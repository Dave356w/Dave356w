#!/usr/bin/env python3
"""Match historical closing moneylines to reconstructed games and grade returns.

One unit is risked per priced selection. Closing-price returns are retrospective
benchmarks, not a claim that a reconstructed pregame decision was executable.
No quote or outcome enters the model's prediction or calibration calculations.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import time
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
import requests

import market_backfill as mb
from fetch_headers import FETCH_HEADERS

DEFAULT_ARCHIVE = Path(__file__).resolve().parent / 'data' / 'mlb_lean_ledger.csv'
MARKETS = ('full', 'f5')
PRICE_COLUMNS = {'full': ('close_away_ml', 'close_home_ml'),
                 'f5': ('f5_close_away_ml', 'f5_close_home_ml')}
# Fetched books in preference order. Books have different holds, so ROI is
# also reported per source rather than only pooled.
ESPN_BOOKS = ((mb.ESPN_DRAFTKINGS, 'espn_draftkings_close'),
              (mb.ESPN_BET, 'espn_espnbet_close'))
ALIASES = {'AZ': 'ARI', 'CHW': 'CWS', 'CHA': 'CWS', 'OAK': 'ATH',
           'SDP': 'SD', 'SFG': 'SF', 'KCR': 'KC', 'TBR': 'TB',
           'WSN': 'WSH', 'ANA': 'LAA'}


def number(value):
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def moneyline(value):
    value = number(value)
    return value if value is not None and abs(value) >= 100 else None


def game_id(row):
    ids = [number(row.get(c)) for c in ('game_pk', 'gamePk')]
    ids = [v for v in ids if v is not None]
    if not ids or any(v <= 0 or v % 1 for v in ids) or len(set(ids)) != 1:
        return None
    return int(ids[0])


def team(value):
    value = str(value).strip().upper()
    return ALIASES.get(value, value)


def identity_matches(prediction, quote):
    if game_id(prediction) is None or game_id(prediction) != game_id(quote):
        return False
    if str(prediction.get('game_date')) != str(quote.get('game_date')):
        return False
    if any(team(prediction.get(s)) != team(quote.get(s)) for s in ('away', 'home')):
        return False
    for side in ('away', 'home'):
        a, b = number(prediction.get('full_' + side)), number(quote.get('full_' + side))
        if a is None or b is None or a != b or a < 0 or a % 1:
            return False
    return True


def initialize(frame):
    required = {'game_pk', 'game_date', 'away', 'home', 'full_away', 'full_home',
                'f5_away', 'f5_home', 'status', 'xw_lean'}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError('Predictions require nonempty game, identity, score and selection columns')
    out = frame.copy(deep=True)
    if out[['game_date', 'away', 'home']].isna().any().any() or any(
            team(r['away']) == team(r['home']) for r in out.to_dict('records')):
        raise ValueError('Predictions require a date and distinct identified teams')
    ids = out.apply(game_id, axis=1)
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError('Predictions require unique, consistent MLB game IDs')
    for market in MARKETS:
        for name in ('home_ml', 'away_ml', 'selection_ml', 'profit_units', 'stake_units',
                     'home_profit_units', 'chalk_profit_units'):
            out[f'{market}_{name}'] = np.nan
        for name in ('odds_status', 'roi_status', 'market_source', 'espn_id',
                     'match_basis', 'settlement'):
            out[f'{market}_{name}'] = pd.Series(None, index=out.index, dtype=object)
        out[f'{market}_odds_status'] = 'no_archive_game'
    out['price_basis'] = 'closing'
    out['f5_settlement_rule'] = 'two_way_ties_push'
    return out


def attach_archive(predictions, archive, source='archive'):
    """Strict ID join; duplicates/conflicts never expand the prediction rows."""
    out = initialize(predictions)
    groups = {}
    for quote in archive.to_dict('records'):
        if quote.get('status') in ('void', 'pending', 'excluded', 'abstained') and not any(
                pd.notna(quote.get(c)) for pair in PRICE_COLUMNS.values() for c in pair):
            # A postponed attempt can share an MLB ID with the eventual game.
            # An unpriced void record is not a competing odds observation.
            continue
        pk = game_id(quote)
        if pk is not None:
            groups.setdefault(pk, []).append(quote)
        else:
            # Conflicting dual ID columns must not become an apparent missing
            # game that a later fallback could silently replace.
            for col in ('game_pk', 'gamePk'):
                value = number(quote.get(col))
                if value is not None and value > 0 and value % 1 == 0:
                    groups.setdefault(int(value), []).append(quote)
    for idx, row in out.iterrows():
        candidates = groups.get(game_id(row), [])
        if not candidates:
            continue
        if not all(identity_matches(row, quote) for quote in candidates):
            for market in MARKETS:
                out.at[idx, f'{market}_odds_status'] = 'identity_mismatch'
            continue
        for market, columns in PRICE_COLUMNS.items():
            status = f'{market}_odds_status'
            # An archive's F5 actuals are an additional consistency check when
            # present; an unrelated/missing fifth inning cannot reject FG odds.
            if market == 'f5' and any(
                number(q.get(f'f5_{s}')) is not None
                and number(row.get(f'f5_{s}')) is not None
                and number(q.get(f'f5_{s}')) != number(row.get(f'f5_{s}'))
                for q in candidates for s in ('away', 'home')):
                out.at[idx, status] = 'f5_score_mismatch'
                continue
            pairs = [(moneyline(q.get(columns[0])), moneyline(q.get(columns[1])))
                     for q in candidates]
            present = [p for p in pairs if None not in p]
            if not present:
                any_value = any(pd.notna(q.get(c)) for q in candidates for c in columns)
                out.at[idx, status] = 'invalid_or_partial_price' if any_value else 'market_unavailable'
                continue
            if len(set(present)) != 1:
                out.at[idx, status] = 'conflicting_prices'
                continue
            if any(any(value is not None and value != present[0][side]
                       for side, value in enumerate(pair)) for pair in pairs):
                out.at[idx, status] = 'conflicting_prices'
                continue
            # Identical duplicates are harmless; complementary partial quotes
            # are deliberately not combined into a manufactured two-sided book.
            away, home = present[0]
            out.at[idx, f'{market}_away_ml'], out.at[idx, f'{market}_home_ml'] = away, home
            out.at[idx, status] = 'matched'
            out.at[idx, f'{market}_market_source'] = source
            out.at[idx, f'{market}_match_basis'] = 'game_id_date_teams_final_score'
            eid = next((q.get('espn_id') for q in candidates if pd.notna(q.get('espn_id'))), None)
            out.at[idx, f'{market}_espn_id'] = str(eid) if eid is not None else None
    return out


def outcome(row, selection, market):
    if selection not in ('away', 'home'):
        return None
    a, h = number(row.get(f'{market}_away')), number(row.get(f'{market}_home'))
    if a is None or h is None or min(a, h) < 0 or a % 1 or h % 1:
        return None
    if a == h:
        return 'T' if market == 'f5' else None
    return 'W' if (h > a) == (selection == 'home') else 'L'


def profit(ml, settlement):
    ml = moneyline(ml)
    if ml is None or settlement not in ('W', 'L', 'T'):
        return None
    return 0.0 if settlement == 'T' else -1.0 if settlement == 'L' else (
        ml / 100 if ml > 0 else 100 / -ml)


def settle(frame):
    """Price each model lean after prediction, never reselect it using odds."""
    out = frame.copy(deep=True)
    for idx, row in out.iterrows():
        selected = next((s for s in ('away', 'home')
                         if team(row.get('xw_lean')) == team(row.get(s))), None)
        for market in MARKETS:
            reason = None
            if row.get('status') != 'graded' or selected is None:
                reason = 'no_graded_selection'
            settlement = outcome(row, selected, market)
            if reason is None and settlement is None:
                reason = 'unsettled_or_incomplete_score'
            saved = row.get(f'xw_{market}')
            if reason is None and pd.notna(saved) and saved != settlement:
                reason = 'prediction_grade_mismatch'
            if reason is None and row.get(f'{market}_odds_status') != 'matched':
                reason = row.get(f'{market}_odds_status')
            if reason is not None:
                out.at[idx, f'{market}_roi_status'] = reason
                continue
            a, h = moneyline(row.get(f'{market}_away_ml')), moneyline(row.get(f'{market}_home_ml'))
            if a is None or h is None:
                out.at[idx, f'{market}_roi_status'] = 'invalid_or_partial_price'
                continue
            chosen = h if selected == 'home' else a
            # Posted implied probabilities choose the favourite. An exact
            # pick'em breaks to home, matching the project's same-row control.
            chalk = 'home' if mb._imp(h) >= mb._imp(a) else 'away'
            out.at[idx, f'{market}_selection_ml'] = chosen
            out.at[idx, f'{market}_settlement'] = settlement
            out.at[idx, f'{market}_profit_units'] = profit(chosen, settlement)
            out.at[idx, f'{market}_stake_units'] = 1.0
            out.at[idx, f'{market}_home_profit_units'] = profit(h, outcome(row, 'home', market))
            out.at[idx, f'{market}_chalk_profit_units'] = profit(h if chalk == 'home' else a,
                                                               outcome(row, chalk, market))
            out.at[idx, f'{market}_roi_status'] = 'graded'
    return out


def summarize(frame):
    result = {'price_basis': 'closing', 'stake_rule': 'risk_one_unit_per_priced_selection',
              'push_rule': 'zero_profit_one_unit_turnover', 'markets': {}}
    for market in MARKETS:
        status = frame.get(f'{market}_roi_status', pd.Series('not_attached', index=frame.index))
        scored = frame[status == 'graded']
        stake = float(pd.to_numeric(scored.get(f'{market}_stake_units', pd.Series(dtype=float))).sum())
        units = float(pd.to_numeric(scored.get(f'{market}_profit_units', pd.Series(dtype=float))).sum())
        settlements = Counter(scored.get(f'{market}_settlement', pd.Series(dtype=object)))
        controls = {}
        for control in ('home', 'chalk'):
            value = float(pd.to_numeric(scored.get(f'{market}_{control}_profit_units',
                                                   pd.Series(dtype=float))).sum())
            controls['always_' + control] = {'games': len(scored), 'profit_units': value,
                                             'roi': value / stake if stake else None}
        result['markets'][market] = {'scheduled_games': len(frame),
            'model_selections': int((frame.get('status', pd.Series(index=frame.index, dtype=object)) == 'graded').sum()),
            'matched_prices': int((frame.get(f'{market}_odds_status', pd.Series(dtype=object)) == 'matched').sum()),
            'graded_games': len(scored), 'wins': settlements['W'], 'losses': settlements['L'],
            'pushes': settlements['T'], 'stake_units': stake, 'profit_units': units,
            'roi': units / stake if stake else None, 'controls': controls,
            'exclusions': dict(Counter(status[status != 'graded'].dropna())),
            'sources': dict(Counter(scored.get(f'{market}_market_source', pd.Series(dtype=object)))),
            'by_source': source_summary(scored, market)}
    return result


def source_summary(scored, market):
    """Model and same-row control ROI within each price source."""
    out = {}
    sources = scored.get(f'{market}_market_source', pd.Series(index=scored.index, dtype=object))
    for source, rows in scored.groupby(sources, sort=True):
        stake = float(pd.to_numeric(rows[f'{market}_stake_units']).sum())
        units = float(pd.to_numeric(rows[f'{market}_profit_units']).sum())
        settlements = Counter(rows[f'{market}_settlement'])
        out[str(source)] = {'graded_games': len(rows), 'wins': settlements['W'],
            'losses': settlements['L'], 'pushes': settlements['T'], 'profit_units': units,
            'roi': units / stake if stake else None,
            **{f'always_{c}_roi': (float(pd.to_numeric(rows[f'{market}_{c}_profit_units']).sum()) / stake
                                   if stake else None) for c in ('home', 'chalk')}}
    return out


class CachedMarketAPI:
    def __init__(self, root, offline=False):
        self.root, self.offline = Path(root), offline

    def __call__(self, url):
        parsed = urlsplit(url)
        if parsed.scheme != 'https' or parsed.hostname not in (
                'statsapi.mlb.com', 'site.api.espn.com', 'sports.core.api.espn.com'):
            raise ValueError('Unexpected odds source host')
        path = self.root / (hashlib.sha256(url.encode()).hexdigest() + '.json')
        if path.exists():
            data = json.loads(path.read_text())
            if data.get('url') != url:
                raise ValueError('Odds cache identity mismatch')
            if data.get('status', 200) != 200:
                raise ValueError('Historical odds endpoint unavailable')
            return data['response']
        if self.offline:
            raise ValueError('Historical odds response not cached')
        for attempt in range(3):
            try:
                r = requests.get(url, headers=FETCH_HEADERS, timeout=(10, 30))
                if r.status_code == 404:
                    self._save(path, url, None, 404)
                r.raise_for_status()
                data = r.json()
                self._save(path, url, data, 200)
                return data
            except (requests.RequestException, ValueError):
                if attempt == 2 or ('r' in locals() and r.status_code == 404):
                    raise
                time.sleep(.5 * 2 ** attempt)

    def _save(self, path, url, response, status):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps({'url': url, 'status': status, 'response': response,
            'fetched_utc': datetime.now(timezone.utc).isoformat()}))
        tmp.replace(path)


@contextmanager
def market_transport(get):
    original = mb._get
    try:
        mb._get = get
        yield
    finally:
        mb._get = original


def verified_event(row, schedule, events):
    """Verify MLB ID before linking ESPN; identical-score DHs need exact start."""
    games = [g for g in schedule if number(g.get('gamePk')) == game_id(row)]
    if len(games) != 1:
        raise ValueError('mlb_game_id_unverified')
    game = games[0]
    quote = dict(game_pk=game['gamePk'], game_date=row['game_date'],
        away=game['away'], home=game['home'], full_away=game['away_score'], full_home=game['home_score'])
    if not identity_matches(row, quote):
        raise ValueError('mlb_identity_mismatch')
    pair = (mb.LEDGER2SA.get(team(row['away']), team(row['away'])),
            mb.LEDGER2SA.get(team(row['home']), team(row['home'])))
    candidates = [e for e in events.get(pair, [])
                  if number(e.get('away_sc')) == number(row.get('full_away'))
                  and number(e.get('home_sc')) == number(row.get('full_home'))]
    if len(candidates) > 1:
        def stamp(value):
            return pd.to_datetime(value, utc=True, errors='coerce')
        start = stamp(game.get('gameDate'))
        candidates = [e for e in candidates if pd.notna(start) and stamp(e.get('start')) == start]
    if len(candidates) != 1:
        raise ValueError('espn_event_ambiguous_or_score_mismatch')
    if candidates[0].get('state') != 'post':
        raise ValueError('espn_event_not_settled')
    return candidates[0]


def backfill_missing(frame, get):
    out = frame.copy(deep=True)
    errors = {}
    with market_transport(get):
        for day, rows in out.groupby('game_date', sort=True):
            eligible = rows[(rows['status'] == 'graded') & rows['xw_lean'].notna()]
            eligible = eligible[eligible[[f'{m}_odds_status' for m in MARKETS]].isin(
                ['no_archive_game', 'market_unavailable']).any(axis=1)]
            if eligible.empty:
                continue
            try:
                schedule, events = mb._statsapi_day(day), mb._espn_day(day)
            except Exception as exc:
                for idx in eligible.index:
                    errors[str(game_id(out.loc[idx]))] = 'schedule_or_event_fetch_failed:' + type(exc).__name__
                continue
            for idx, row in eligible.iterrows():
                try:
                    event = verified_event(row, schedule, events)
                except Exception as exc:
                    errors[str(game_id(row))] = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
                    continue
                for market in MARKETS:
                    if row[f'{market}_odds_status'] not in ('no_archive_game', 'market_unavailable'):
                        continue
                    try:
                        # DraftKings first; ESPN BET only when ESPN lists no
                        # two-sided DraftKings close. Each keeps its own label.
                        for provider, source in ESPN_BOOKS:
                            values = (mb._espn_close(event['eid'], provider) if market == 'full'
                                      else mb._espn_f5(event['eid'], event.get('tid') or {}, provider))
                            a, h = (moneyline((values or {}).get(c)) for c in PRICE_COLUMNS[market])
                            if a is not None and h is not None:
                                break
                        if a is None or h is None:
                            out.at[idx, f'{market}_odds_status'] = 'market_unavailable'
                            continue
                        out.at[idx, f'{market}_away_ml'], out.at[idx, f'{market}_home_ml'] = a, h
                        out.at[idx, f'{market}_odds_status'] = 'matched'
                        out.at[idx, f'{market}_market_source'] = source
                        out.at[idx, f'{market}_espn_id'] = str(event['eid'])
                        out.at[idx, f'{market}_match_basis'] = 'mlb_id_date_teams_scores_espn_settled'
                    except Exception as exc:
                        errors[f'{game_id(row)}:{market}'] = type(exc).__name__
    return out, errors


def grade(predictions, archive=None, source='archive', fetch_missing=False, get=None):
    frame = attach_archive(predictions, archive if archive is not None else pd.DataFrame(), source)
    errors = {}
    if fetch_missing:
        if get is None:
            raise ValueError('A cached odds transport is required')
        frame, errors = backfill_missing(frame, get)
    frame = settle(frame)
    result = summarize(frame)
    result['fetch_errors'] = errors
    result['generated_utc'] = datetime.now(timezone.utc).isoformat()
    result['source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                               for name in ('replay_market.py', 'market_backfill.py')}
    return frame, result


def render_panel(frame):
    if 'full_roi_status' not in frame:
        return ''
    summary = summarize(frame)
    rows = []
    def fmt(value):
        return '—' if value is None else f'{100 * value:+.2f}%'
    for market, title in (('full', 'Full game'), ('f5', 'First five')):
        s = summary['markets'][market]
        rows.append(f"<tr><td>{title}</td><td>{s['graded_games']} / {s['model_selections']}</td>"
            f"<td>{s['wins']}–{s['losses']}–{s['pushes']}</td><td>{s['profit_units']:+.3f}u</td>"
            f"<td>{fmt(s['roi'])}</td><td>{fmt(s['controls']['always_chalk']['roi'])}</td>"
            f"<td>{fmt(s['controls']['always_home']['roi'])}</td></tr>")
        if len(s['by_source']) > 1:
            for source, b in s['by_source'].items():
                rows.append(f"<tr><td>&nbsp;· {html.escape(source)}</td><td>{b['graded_games']}</td>"
                    f"<td>{b['wins']}–{b['losses']}–{b['pushes']}</td><td>{b['profit_units']:+.3f}u</td>"
                    f"<td>{fmt(b['roi'])}</td><td>{fmt(b['always_chalk_roi'])}</td>"
                    f"<td>{fmt(b['always_home_roi'])}</td></tr>")
    exclusions = ''.join(f'<li>{html.escape(m)} · {html.escape(k)}: {v}</li>'
        for m, s in summary['markets'].items() for k, v in s['exclusions'].items())
    return ('<h2>Closing-price retrospective ROI</h2><p class="note">One unit risked per priced selection. '
        'ROI = net profit / total units staked; F5 ties push (zero profit, one unit turnover). '
        'Missing or rejected prices are excluded. Controls use exactly the same games and market. '
        'Closing prices are evaluation benchmarks; they are not proof of executable pregame returns. '
        'No pregame/closing or full-game/F5 price fallback is used. Exact pick’em breaks to home. '
        'Fetched closes are DraftKings; ESPN BET is used only where ESPN lists no DraftKings close.</p>'
        '<div class="scroll"><table><thead><tr><th>Market</th><th>Priced / selections</th>'
        '<th>W–L–Push</th><th>Profit</th><th>Model ROI</th><th>Chalk ROI</th><th>Home ROI</th>'
        '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
        '<details><summary>ROI exclusions</summary><ul>' + (exclusions or '<li>None</li>') + '</ul></details>')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True)
    p.add_argument('--odds-csv', default=str(DEFAULT_ARCHIVE))
    p.add_argument('--fetch-missing', action='store_true')
    p.add_argument('--cache', default='replay_cache/odds')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--output', default='replay_market_output')
    a = p.parse_args()
    import historical_replay as hr
    output, cache = hr.safe_output(a.output), hr.safe_output(a.cache)
    path = Path(a.predictions)
    raw = path.read_bytes()
    meta_path = path.with_name('manifest.json')
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    expected = meta.get('output_sha256', {}).get('predictions.csv')
    if expected and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Prediction file does not match its run manifest')
    predictions = pd.read_csv(path)
    archive = pd.read_csv(a.odds_csv)
    frame, market = grade(predictions, archive, Path(a.odds_csv).name, a.fetch_missing,
                          CachedMarketAPI(cache, a.offline))
    meta.update(start=str(frame.game_date.min()), end=str(frame.game_date.max()),
        model_tag=str(frame.model_tag.iloc[0]), untracked_bip=meta.get('untracked_bip', 0),
        source_commit=meta.get('source_commit', 'external_predictions'),
        input_prediction_sha256=hashlib.sha256(raw).hexdigest(), market_summary=market,
        odds_archive_sha256=hashlib.sha256(Path(a.odds_csv).read_bytes()).hexdigest())
    hr.write_report(frame, meta, output)
    print(json.dumps(market, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
