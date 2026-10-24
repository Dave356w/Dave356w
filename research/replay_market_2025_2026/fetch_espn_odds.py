#!/usr/bin/env python3
"""Rebuild espn_full_odds.csv and espn_f5_odds.csv from ESPN's public odds API.

For every graded replay game it records each ESPN provider's full-game
opening/closing moneylines, and the replay book's "1st 5 Innings Moneyline"
(propBets type 136) value and opener per team. 2025 events carry ESPN BET
(58, plus live feed 59); 2026 events carry DraftKings (100).

    python research/replay_market_2025_2026/fetch_espn_odds.py

Responses are fetched fresh; ESPN may revise them, so the committed CSVs are
the snapshot the analysis used (fetched 2026-10-23/24 UTC).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
API = 'https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb'
SEASONS = {'2025': ('replay_2025', '58'), '2026': ('replay_2026', '100')}


def amer(value):
    try:
        return int(str(value).replace('+', ''))
    except (TypeError, ValueError):
        return None


def get(url):
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(1 + 2 * attempt)


def full_odds(job):
    season, eid = job
    rows = []
    for item in (get(f'{API}/events/{eid}/competitions/{eid}/odds') or {}).get('items', []):
        row = {'season': season, 'espn_id': eid, 'provider_id': str(item['provider']['id'])}
        for kind in ('open', 'close'):
            for side in ('home', 'away'):
                quote = ((item.get(f'{side}TeamOdds') or {}).get(kind) or {}).get('moneyLine') or {}
                row[f'{kind}_{side}'] = amer(quote.get('american'))
        rows.append(row)
    return rows


def f5_odds(job, teams):
    season, eid, provider = job
    base = f'{API}/events/{eid}/competitions/{eid}/odds/{provider}/propBets'
    page = get(base + '?limit=200') or {}
    items = list(page.get('items', []))
    for n in range(2, int(page.get('pageCount', 1)) + 1):
        items += (get(base + f'?limit=200&page={n}') or {}).get('items', [])
    rows = []
    for item in items:
        if str((item.get('type') or {}).get('id')) != '136':
            continue
        team_id = ((item.get('team') or {}).get('$ref') or '').rsplit('/', 1)[-1].split('?')[0]
        odds = (item.get('odds') or {}).get('american') or {}
        rows.append({'season': season, 'espn_id': eid, 'provider_id': provider,
                     'espn_team_id': team_id, 'team': teams.get(team_id),
                     'close_ml': amer(odds.get('value')), 'open_ml': amer(odds.get('open'))})
    return rows


def main():
    teams = pd.read_csv(HERE / 'espn_team_ids.csv', dtype={'espn_team_id': str})
    teams = teams.set_index('espn_team_id').team.to_dict()
    full_jobs, f5_jobs = [], []
    for season, (folder, provider) in SEASONS.items():
        d = pd.read_csv(HERE / folder / 'predictions.csv', low_memory=False)
        full_jobs += [(season, int(float(e))) for e in d[d.full_profit_units.notna()].full_espn_id.unique()]
        f5_jobs += [(season, int(float(e)), provider) for e in d[d.f5_profit_units.notna()].f5_espn_id.unique()]
    with ThreadPoolExecutor(8) as pool:
        full = [r for rows in pool.map(full_odds, full_jobs) for r in rows]
        f5 = [r for rows in pool.map(lambda j: f5_odds(j, teams), f5_jobs) for r in rows]
    pd.DataFrame(full).to_csv(HERE / 'espn_full_odds.csv', index=False)
    pd.DataFrame(f5).to_csv(HERE / 'espn_f5_odds.csv', index=False)
    print(f'{len(full_jobs)} full-game events, {len(f5_jobs)} F5 events')


if __name__ == '__main__':
    main()
