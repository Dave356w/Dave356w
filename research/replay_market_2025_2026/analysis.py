#!/usr/bin/env python3
"""Market-relative audit of the v14 full-season replays (2025, 2026).

Reads only committed inputs in this folder plus data/mlb_lean_ledger.csv and
prints every table in RESULTS.txt:

    python research/replay_market_2025_2026/analysis.py > research/replay_market_2025_2026/RESULTS.txt

Replay rows are hindsight (final lineups/starters, model fixed today); the
native ledger section is the only pregame-locked evidence. Uncertainties are
±1 SE, clustered by slate date. No model, ledger or registered rule changes.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEASONS = {'2025': ('replay_2025', '58', 'ESPN BET'), '2026': ('replay_2026', '100', 'DraftKings')}
BANDS = [-1e9, -250, -175, -130, -99.5, 129.5, 174.5, 249.5, 1e9]
BAND_LABELS = ['<=-250', '-249..-175', '-174..-130', '-129..-100',
               '+100..+129', '+130..+174', '+175..+249', '>=+250']
PER = 0.020  # delta coefficients are per 0.020 xwOBA of xw_net (home-signed)

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 30)
pd.set_option('display.float_format', '{:.1f}'.format)


def imp(ml):
    ml = np.asarray(ml, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))


def novig(home_ml, away_ml):
    h, a = imp(home_ml), imp(away_ml)
    return h / (h + a)


def logit(p):
    return np.log(p / (1 - p))


def logloss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def payout(ml, won):
    ml = np.asarray(ml, float)
    return np.where(won == 1, np.where(ml > 0, ml / 100, 100 / -ml), -1.0)


def cluster_se(values, dates):
    frame = pd.DataFrame({'v': np.asarray(values, float), 'd': np.asarray(dates)})
    sums = frame.groupby('d').v.sum()
    return np.sqrt(len(sums)) * sums.std(ddof=1) / len(frame)


def ols_cluster(x, y, dates):
    X = np.c_[np.ones(len(x)), x]
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)
    meat = np.zeros((2, 2))
    for _, idx in pd.Series(range(len(x))).groupby(np.asarray(dates)):
        u = (X[idx.values] * e[idx.values, None]).sum(0)
        meat += np.outer(u, u)
    return b, np.sqrt(np.diag(bread @ meat @ bread))


def offset_logit(x, offset, y):
    """Fit logit(p) = offset + b·x; return b and its model SE."""
    b = 0.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-(offset + b * x)))
        h = (x * x * p * (1 - p)).sum()
        b += (x * (y - p)).sum() / h
    return b, 1 / np.sqrt(h)


def load(season, market):
    folder, provider, book = SEASONS[season]
    d = pd.read_csv(HERE / folder / 'predictions.csv', low_memory=False)
    g = d[d[f'{market}_profit_units'].notna()].copy()
    lean_home = (g.xw_lean == g.home).values
    sel, oth = g[f'{market}_selection_ml'].values, np.where(
        lean_home, g[f'{market}_away_ml'], g[f'{market}_home_ml'])
    g['q'] = imp(sel) / (imp(sel) + imp(oth))
    g['be'] = imp(sel)
    g['pr'] = g[f'{market}_profit_units']
    g['decided'] = g.pr != 0
    g['win'] = (g.pr > 0).astype(float)
    g['hold'] = imp(g[f'{market}_home_ml']) + imp(g[f'{market}_away_ml']) - 1
    g['month'] = pd.to_datetime(g.game_date).dt.month.clip(lower=4)
    g['band'] = pd.cut(sel, BANDS, labels=BAND_LABELS)
    g['lean_home'] = lean_home
    g['x'] = g.xw_net / PER
    return g


def summary_row(x):
    y = x[x.decided]
    e = y.win - y.q
    return pd.Series({
        'n': len(x), 'win%': 100 * y.win.mean(), 'q%': 100 * y.q.mean(),
        'excess_pp': 100 * e.mean(), 'se': 100 * cluster_se(e, y.game_date) if len(y) > 2 else np.nan,
        'units': x.pr.sum(), 'roi%': 100 * x.pr.mean(),
        'roi_se': 100 * cluster_se(x.pr, x.game_date) if len(x) > 2 else np.nan,
        'mkt_null%': 100 * (y.q / y.be - 1).mean(),
        'chalk_roi%': 100 * x[f'{MARKET}_chalk_profit_units'].mean(),
        'home_roi%': 100 * x[f'{MARKET}_home_profit_units'].mean(),
        'hold%': 100 * x.hold.mean()})


def section(title):
    print('\n' + '=' * 78 + f'\n{title}\n' + '=' * 78)


def provenance():
    section('1. Provenance')
    for season, (folder, provider, book) in SEASONS.items():
        m = json.loads((HERE / folder / 'manifest.json').read_text())
        mk = m['market_summary']['markets']
        print(f"{season}: model_tag={m['model_tag']} source_commit={m['source_commit'][:7]} "
              f"window={m['start']}..{m['end']} scheduled={m['summary']['scheduled_games']} "
              f"selections={m['summary']['predicted_games']} | full sources={mk['full']['sources']} "
              f"| f5 sources={mk['f5']['sources']}")


def season_tables():
    global MARKET
    for MARKET in ('full', 'f5'):
        section(f'2. Season and month at the CLOSE — {MARKET}')
        for season in SEASONS:
            g = load(season, MARKET)
            t = g.groupby('month').apply(summary_row, include_groups=False)
            t.loc['season'] = summary_row(g)
            note = '; ledger closes from Jul 2' if season == '2026' else ''
            print(f'-- {season} ({SEASONS[season][2]} closes via ESPN{note})')
            print(t)
        section(f'3. Closing price band of the model side — {MARKET} (descriptive, not a filter)')
        for season in SEASONS:
            g = load(season, MARKET)
            t = g.groupby('band', observed=True).apply(summary_row, include_groups=False)
            t.loc['pooled'] = summary_row(g)
            print(f'-- {season}')
            print(t[['n', 'win%', 'q%', 'excess_pp', 'se', 'units', 'roi%', 'roi_se', 'mkt_null%']])


def hold_anomaly():
    section('4. Closing hold by week, 2026 full game (ESPN DraftKings close)')
    g = load('2026', 'full')
    week = pd.to_datetime(g.game_date).dt.to_period('W')
    print((100 * g.groupby(week).hold.mean()).to_string(float_format='{:.2f}'.format))


def delta_vs_close():
    section('5. Does the delta add to the CLOSING market? (held-out log loss)')
    for market in ('full', 'f5'):
        data = {}
        for season in SEASONS:
            g = load(season, market)
            g = g[g[f'{market}_home'] != g[f'{market}_away']].copy()
            g['lq'] = logit(novig(g[f'{market}_home_ml'], g[f'{market}_away_ml']))
            g['y'] = (g[f'{market}_home'] > g[f'{market}_away']).astype(float)
            data[season] = g
            b, se = offset_logit(g.x.values, g.lq.values, g.y.values)
            print(f'{market} {season}: delta coefficient given close {b:+.3f} ± {se:.3f} (n={len(g)})')
        for tr, te in (('2025', '2026'), ('2026', '2025')):
            b, _ = offset_logit(data[tr].x.values, data[tr].lq.values, data[tr].y.values)
            B = data[te]
            p0 = 1 / (1 + np.exp(-B.lq.values))
            p1 = 1 / (1 + np.exp(-(B.lq.values + b * B.x.values)))
            dl = logloss(p1, B.y.values) - logloss(p0, B.y.values)
            print(f'{market} fit {tr} -> test {te}: Δlogloss vs close {dl.mean():+.5f} ± '
                  f'{cluster_se(dl, B.game_date):.5f} (positive = worse than market)')


def open_frame(season, market):
    g = load(season, market)
    provider = SEASONS[season][1]
    if market == 'full':
        o = pd.read_csv(HERE / 'espn_full_odds.csv', dtype={'provider_id': str})
        o = o[(o.season == int(season)) & (o.provider_id == provider)]
        g['espn_id'] = g.full_espn_id.astype(float).astype(int)
        g = g.merge(o[['espn_id', 'open_home', 'open_away', 'close_home', 'close_away']],
                    on='espn_id', how='left', validate='one_to_one')
        g['open_home_ml'], g['open_away_ml'] = g.open_home, g.open_away
    else:
        o = pd.read_csv(HERE / 'espn_f5_odds.csv', dtype={'provider_id': str, 'espn_team_id': str})
        o = o[(o.season == int(season)) & (o.provider_id == provider)]
        g['espn_id'] = g.f5_espn_id.astype(float).astype(int)
        for side in ('home', 'away'):
            s = o.rename(columns={'team': side, 'open_ml': f'open_{side}_ml',
                                  'close_ml': f'espn_close_{side}'})[['espn_id', side, f'open_{side}_ml', f'espn_close_{side}']]
            g = g.merge(s, on=['espn_id', side], how='left', validate='one_to_one')
        # Orientation check: ESPN's close for each side must equal the graded close.
        bad = (g.espn_close_home != g.f5_home_ml) | (g.espn_close_away != g.f5_away_ml)
        g.loc[bad, ['open_home_ml', 'open_away_ml']] = np.nan
    g = g[g[['open_home_ml', 'open_away_ml']].notna().all(1)
          & (g[f'{market}_home'] != g[f'{market}_away'])].copy()
    g['qo'] = novig(g.open_home_ml, g.open_away_ml)
    g['qc'] = novig(g[f'{market}_home_ml'], g[f'{market}_away_ml'])
    g['y'] = (g[f'{market}_home'] > g[f'{market}_away']).astype(float)
    return g


def open_tests():
    section('6. Opening price: line movement, CLV and results at the OPEN')
    print('Replay deltas use final lineups/starters; the open usually predates them.')
    for market in ('full', 'f5'):
        data = {}
        for season in SEASONS:
            g = open_frame(season, market)
            data[season] = g
            move = logit(g.qc) - logit(g.qo)
            b, se = ols_cluster(g.x.values, move.values, g.game_date.values)
            s = np.where(g.lean_home, 1, -1)
            clv = s * (g.qc - g.qo)
            bo, so = offset_logit(g.x.values, logit(g.qo).values, g.y.values)
            won = np.where(g.lean_home, g.y, 1 - g.y)
            ml = np.where(g.lean_home, g.open_home_ml, g.open_away_ml)
            other = np.where(g.lean_home, g.open_away_ml, g.open_home_ml)
            q = imp(ml) / (imp(ml) + imp(other))
            pr = payout(ml, won)
            moved = move != 0
            print(f'{market} {season} n={len(g)}: move per {PER:.3f} {b[1]:+.4f} ± {se[1]:.4f}; '
                  f'CLV {100 * clv.mean():+.2f} ± {100 * cluster_se(clv, g.game_date):.2f} pp; '
                  f'toward model {100 * (np.sign(move) == s)[moved].mean():.1f}% of {moved.sum()} moved; '
                  f'delta coef given open {bo:+.3f} ± {so:.3f}')
            print(f'    at OPEN: excess {100 * (won - q).mean():+.1f} ± {100 * cluster_se(won - q, g.game_date):.1f} pp; '
                  f'ROI {100 * pr.mean():+.1f} ± {100 * cluster_se(pr, g.game_date):.1f}% '
                  f'(market-correct null {100 * (q / imp(ml) - 1).mean():+.1f}%)')
        for tr, te in (('2025', '2026'), ('2026', '2025')):
            A, B = data[tr], data[te]
            b, _ = offset_logit(A.x.values, logit(A.qo).values, A.y.values)
            p1 = 1 / (1 + np.exp(-(logit(B.qo).values + b * B.x.values)))
            dl = logloss(p1, B.y.values) - logloss(B.qo.values, B.y.values)
            print(f'{market} fit {tr} -> test {te}: Δlogloss vs open {dl.mean():+.5f} ± {cluster_se(dl, B.game_date):.5f}')


def native_lock():
    section('7. Native ledger: lock-time price -> close (pregame-locked evidence)')
    L = pd.read_csv(ROOT / 'data' / 'mlb_lean_ledger.csv', low_memory=False)
    cols = ['pregame_home_ml', 'pregame_away_ml', 'close_home_ml', 'close_away_ml']
    g = L[(L.status == 'graded') & L.xw_net.notna() & L[cols].notna().all(1)].copy()
    g['qp'] = novig(g.pregame_home_ml, g.pregame_away_ml)
    g['qc'] = novig(g.close_home_ml, g.close_away_ml)
    g['x'] = g.xw_net / PER
    g['move'] = logit(g.qc) - logit(g.qp)
    g['family'] = np.where(g.model_tag.isin(['xw+starter_velo_v14', 'xw+starter_blend_v13']),
                           'v13-v14', 'earlier tags')
    print(f'{len(g)} graded rows, {g.game_date.min()}..{g.game_date.max()}; lineup_status_home: '
          f'{g.lineup_status_home.value_counts().to_dict()}')
    for label, x in [('all native', g)] + list(g.groupby('family')):
        b, se = ols_cluster(x.x.values, x.move.values, x.game_date.values)
        s = np.where(x.xw_lean == x.home, 1, -1)
        clv = s * (x.qc - x.qp)
        print(f'  {label:12s} n={len(x):4d} slates={x.game_date.nunique():3d}: lock->close move per {PER:.3f} '
              f'{b[1]:+.4f} ± {se[1]:.4f}; CLV {100 * clv.mean():+.2f} ± {100 * cluster_se(clv, x.game_date):.2f} pp')
    h = g[g[['open_home_ml', 'open_away_ml']].notna().all(1)].copy()
    qo = novig(h.open_home_ml, h.open_away_ml)
    for label, move in (('open->lock', logit(h.qp) - logit(qo)), ('open->close', logit(h.qc) - logit(qo))):
        b, se = ols_cluster(h.x.values, move.values, h.game_date.values)
        print(f'  {label:12s} n={len(h)}: move per {PER:.3f} {b[1]:+.4f} ± {se[1]:.4f}')


if __name__ == '__main__':
    MARKET = 'full'
    provenance()
    season_tables()
    hold_anomaly()
    delta_vs_close()
    open_tests()
    native_lock()
