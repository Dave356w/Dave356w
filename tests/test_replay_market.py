"""Odds identity, market isolation, unit-return arithmetic and reproducibility."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

import historical_replay as hr
import replay_market as rm


def row(pk=1, **values):
    r = dict(game_pk=pk, game_date='2026-04-01', away='NYY', home='BOS',
             full_away=2, full_home=3, f5_away=1, f5_home=1,
             status='graded', xw_lean='BOS', xw_full='W', xw_f5='T',
             model_tag='xw+starter_velo_v14', xw_net=.02, exclusion_reason=None)
    r.update(values)
    return r


def odds(pk=1, **values):
    r = row(pk)
    r.update(close_away_ml=120, close_home_ml=-150,
             f5_close_away_ml=110, f5_close_home_ml=-130)
    r.update(values)
    return r


def grade(rows=None, quotes=None):
    return rm.grade(pd.DataFrame(rows or [row()]), pd.DataFrame(quotes if quotes is not None else [odds()]))


@pytest.mark.parametrize('ml,settlement,expected', [
    (150,'W',1.5),(-200,'W',.5),(100,'W',1),(-100,'W',1),
    (-150,'L',-1),(200,'L',-1),(-200,'T',0),(200,'T',0)])
def test_flat_risk_one_unit_arithmetic(ml,settlement,expected):
    assert rm.profit(ml,settlement) == pytest.approx(expected)


@pytest.mark.parametrize('bad', [0,99,-99,None,np.nan,np.inf,-np.inf,'junk',True])
def test_invalid_prices_are_never_inferred_or_zero_filled(bad):
    f,s = grade(quotes=[odds(close_home_ml=bad)])
    assert f.loc[0,'full_roi_status'] == 'invalid_or_partial_price'
    assert pd.isna(f.loc[0,'full_profit_units'])
    assert s['markets']['full']['roi'] is None
    assert s['markets']['full']['stake_units'] == 0
    assert s['markets']['f5']['graded_games'] == 1


def test_f5_push_has_zero_profit_and_counts_in_turnover():
    f,s = grade()
    assert f.loc[0,'full_profit_units'] == pytest.approx(2/3)
    assert f.loc[0,'f5_profit_units'] == 0
    assert f.loc[0,'f5_stake_units'] == 1
    assert s['markets']['f5']['pushes'] == 1
    assert s['markets']['f5']['roi'] == 0


def test_away_selection_gets_away_price_even_when_home_wins():
    f,s = grade([row(xw_lean='NYY',xw_full='L')])
    assert f.loc[0,'full_selection_ml'] == 120
    assert f.loc[0,'full_profit_units'] == -1


@pytest.mark.parametrize('change', [dict(game_date='2026-04-02'),dict(away='BOS',home='NYY'),
    dict(full_away=3),dict(full_home=99),dict(gamePk=999)])
def test_identity_and_scores_are_verified_before_accepting_prices(change):
    f,s = grade(quotes=[odds(**change)])
    assert f.loc[0,'full_odds_status'] == 'identity_mismatch'
    assert s['markets']['full']['graded_games'] == 0


def test_join_is_by_game_id_so_same_day_doubleheaders_do_not_cross():
    p = [row(1),row(2)]
    q = [odds(2,close_home_ml=150), odds(1,close_home_ml=-200)]
    f,_ = grade(p,q)
    assert len(f)==2
    assert list(f.full_selection_ml) == [-200,150]


def test_identical_duplicates_do_not_inflate_stakes_and_conflicts_are_rejected():
    f,s = grade(quotes=[odds(),odds()])
    assert len(f)==1 and s['markets']['full']['stake_units']==1
    f,s = grade(quotes=[odds(),odds(close_home_ml=-180)])
    assert f.loc[0,'full_odds_status']=='conflicting_prices'
    assert s['markets']['full']['roi'] is None
    assert s['markets']['f5']['graded_games']==1


def test_duplicate_prediction_ids_are_rejected():
    with pytest.raises(ValueError,match='unique'):
        grade([row(),row()])


def test_aliases_are_normalized_without_swapping_sides():
    f,_ = grade([row(away='ARI',xw_lean='ARI',xw_full='L')],[odds(away='AZ')])
    assert f.loc[0,'full_odds_status']=='matched'
    assert f.loc[0,'full_selection_ml']==120


def test_pregame_and_full_game_prices_are_not_fallbacks_for_other_markets():
    q=odds()
    for col in rm.PRICE_COLUMNS['full']+rm.PRICE_COLUMNS['f5']:
        q[col]=np.nan
    q.update(pregame_home_ml=-150,pregame_away_ml=130)
    f,s=grade(quotes=[q])
    assert s['markets']['full']['roi'] is None
    q.update(close_home_ml=-150,close_away_ml=130)
    f,s=grade(quotes=[q])
    assert s['markets']['full']['graded_games']==1
    assert s['markets']['f5']['roi'] is None


def test_f5_score_mismatch_does_not_destroy_full_game_match():
    f,s=grade(quotes=[odds(f5_home=9)])
    assert f.loc[0,'f5_odds_status']=='f5_score_mismatch'
    assert s['markets']['full']['graded_games']==1
    assert s['markets']['f5']['graded_games']==0


@pytest.mark.parametrize('change,reason', [
    (dict(status='abstained',xw_lean=None),'no_graded_selection'),
    (dict(status='pending'),'no_graded_selection'),
    (dict(status='void'),'no_graded_selection'),
    (dict(xw_full='L'),'prediction_grade_mismatch'),
    (dict(full_home=None),'unsettled_or_incomplete_score')])
def test_unusable_predictions_never_enter_turnover(change,reason):
    f,s=grade([row(**change)])
    assert f.loc[0,'full_roi_status']==reason
    assert s['markets']['full']['stake_units']==0


def test_controls_and_roi_share_exactly_the_priced_model_rows():
    p=[row(1),row(2,xw_lean='NYY',xw_full='L'),row(3)]
    f,s=grade(p,[odds(1),odds(2,close_home_ml=130,close_away_ml=-140)])
    full=s['markets']['full']
    assert full['graded_games']==2 and full['matched_prices']==2
    assert full['roi']==pytest.approx(((2/3)-1)/2)
    assert full['controls']['always_home']['games']==2
    assert full['controls']['always_chalk']['games']==2
    assert full['controls']['always_home']['profit_units']==pytest.approx(2/3+1.3)
    assert full['controls']['always_chalk']['profit_units']==pytest.approx(2/3-1)
    assert full['exclusions']=={'no_archive_game':1}


def test_rerun_is_idempotent_and_never_changes_model_columns():
    original=pd.DataFrame([row()]);f,s=rm.grade(original,pd.DataFrame([odds()]))
    pd.testing.assert_frame_equal(original,f[original.columns])
    again,s2=rm.grade(f,pd.DataFrame([odds()]))
    pd.testing.assert_frame_equal(f,again)
    assert s['markets']==s2['markets']


def event(eid='e1', **changes):
    e=dict(eid=eid,away_sc='2',home_sc='3',state='post',start='2026-04-01T20:00:00Z',tid={'1':'away','2':'home'})
    e.update(changes)
    return e


def sched(pk=1, **changes):
    g=dict(gamePk=pk,gameDate='2026-04-01T20:00:00Z',away='NYY',home='BOS',away_score=2,home_score=3)
    g.update(changes)
    return g


def test_espn_join_rejects_even_a_single_event_with_the_wrong_score():
    with pytest.raises(ValueError,match='score_mismatch'):
        rm.verified_event(row(),[sched()],{('NYY','BOS'):[event(home_sc='10')]})
    with pytest.raises(ValueError,match='not_settled'):
        rm.verified_event(row(),[sched()],{('NYY','BOS'):[event(state='in')]})
    with pytest.raises(ValueError,match='game_id'):
        rm.verified_event(row(),[sched(2)],{('NYY','BOS'):[event()]})


def test_identical_score_doubleheader_requires_unique_exact_start():
    events={('NYY','BOS'):[event('e1'),event('e2',start='2026-04-02T00:00:00Z')]}
    assert rm.verified_event(row(),[sched()],events)['eid']=='e1'
    events[('NYY','BOS')][1]['start']=events[('NYY','BOS')][0]['start']
    with pytest.raises(ValueError,match='ambiguous'):
        rm.verified_event(row(),[sched()],events)


def test_backfill_reuses_market_parsers_and_restores_transport(monkeypatch):
    original=rm.mb._get
    calls=[]
    monkeypatch.setattr(rm.mb,'_statsapi_day',lambda d:[sched()])
    monkeypatch.setattr(rm.mb,'_espn_day',lambda d:{('NYY','BOS'):[event()]})
    monkeypatch.setattr(rm.mb,'_espn_close',lambda eid,provider=rm.mb.ESPN_DRAFTKINGS:dict(close_away_ml=120,close_home_ml=-150))
    monkeypatch.setattr(rm.mb,'_espn_f5',lambda eid,tid,provider=rm.mb.ESPN_DRAFTKINGS:dict(f5_close_away_ml=110,f5_close_home_ml=-130))
    f,s=rm.grade(pd.DataFrame([row()]),fetch_missing=True,get=lambda url:calls.append(url))
    assert f.loc[0,'full_market_source']=='espn_draftkings_close'
    assert f.loc[0,'full_espn_id']=='e1'
    assert s['markets']['full']['graded_games']==1
    assert rm.mb._get is original


def test_bad_archive_identity_is_not_silently_replaced_by_backfill(monkeypatch):
    def forbidden(*args): raise AssertionError('unexpected fallback')
    monkeypatch.setattr(rm.mb,'_statsapi_day',forbidden)
    f,s=rm.grade(pd.DataFrame([row()]),pd.DataFrame([odds(full_home=99)]),fetch_missing=True,get=forbidden)
    assert f.loc[0,'full_odds_status']=='identity_mismatch'
    assert not s['fetch_errors']


def test_cached_market_transport_reproduces_without_network(tmp_path,monkeypatch):
    class Response:
        status_code=200
        def raise_for_status(self): pass
        def json(self): return {'items':[1]}
    monkeypatch.setattr(rm.requests,'get',lambda *a,**k:Response())
    url='https://sports.core.api.espn.com/v2/sports/baseball/leagues/mlb/events/1/odds'
    result=rm.CachedMarketAPI(tmp_path)(url)
    assert rm.CachedMarketAPI(tmp_path,offline=True)(url)==result
    with pytest.raises(ValueError,match='not cached'):
        rm.CachedMarketAPI(tmp_path,offline=True)(url+'x')
    with pytest.raises(ValueError,match='host'):
        rm.CachedMarketAPI(tmp_path)('https://unknown.example/odds')


def test_report_discloses_price_basis_coverage_and_empty_roi():
    f,s=grade(quotes=[])
    panel=rm.render_panel(f)
    assert 'Closing-price retrospective ROI' in panel
    assert '0 / 1' in panel
    assert '—' in panel
    assert 'zero profit, one unit turnover' in panel
    assert 'ROI = net profit / total units staked' in panel


def test_all_existing_archive_prices_grade_against_independent_vector_math():
    # Real stored decisions of multiple versions: an arithmetic/matching audit,
    # not a claim that these are the new replay's predictions.
    ledger=pd.read_csv(rm.DEFAULT_ARCHIVE)
    # The prediction's MLB identity is primary. A conflicting market-side
    # gamePk in an archived row must be excluded rather than trusted.
    predictions=ledger[ledger.status=='graded'].drop(columns=['gamePk'])
    f,s=rm.grade(predictions,ledger)
    p=predictions.merge(f[['game_pk','full_roi_status']],on='game_pk',validate='one_to_one')
    p=p[p.full_roi_status=='graded']
    home=p.xw_lean==p.home
    ml=np.where(home,p.close_home_ml,p.close_away_ml)
    win=np.where(home,p.full_home>p.full_away,p.full_away>p.full_home)
    expected=np.where(win,np.where(ml>0,ml/100,100/-ml),-1)
    assert len(p)>1000
    assert s['markets']['full']['profit_units']==pytest.approx(float(expected.sum()))
    assert s['markets']['full']['roi']==pytest.approx(float(expected.mean()))


def test_unpriced_postponed_attempt_does_not_conflict_with_eventual_quote():
    q=odds(game_date='2026-03-31',status='void',full_home=None,full_away=None)
    for c in rm.PRICE_COLUMNS['full']+rm.PRICE_COLUMNS['f5']:
        q[c]=np.nan
    f,s=grade(quotes=[q,odds()])
    assert f.loc[0,'full_odds_status']=='matched'
    assert s['markets']['full']['graded_games']==1


def test_conflicting_partial_quote_is_not_ignored():
    f,s=grade(quotes=[odds(),odds(close_away_ml=999,close_home_ml=None)])
    assert f.loc[0,'full_odds_status']=='conflicting_prices'


def test_full_game_fetch_failure_does_not_block_f5(monkeypatch):
    monkeypatch.setattr(rm.mb,'_statsapi_day',lambda d:[sched()])
    monkeypatch.setattr(rm.mb,'_espn_day',lambda d:{('NYY','BOS'):[event()]})
    def fail(eid,provider=None): raise RuntimeError('unavailable')
    monkeypatch.setattr(rm.mb,'_espn_close',fail)
    monkeypatch.setattr(rm.mb,'_espn_f5',lambda eid,tid,provider=rm.mb.ESPN_DRAFTKINGS:dict(f5_close_away_ml=110,f5_close_home_ml=-130))
    f,s=rm.grade(pd.DataFrame([row()]),fetch_missing=True,get=lambda *a:None)
    assert s['markets']['full']['graded_games']==0
    assert s['markets']['f5']['graded_games']==1
    assert s['fetch_errors']=={'1:full':'RuntimeError'}


def test_existing_prediction_cli_outputs_one_consistent_report(tmp_path,monkeypatch):
    inp=tmp_path/'input';inp.mkdir()
    p=inp/'predictions.csv';q=inp/'odds.csv'
    pd.DataFrame([row()]).to_csv(p,index=False)
    pd.DataFrame([odds()]).to_csv(q,index=False)
    out=tmp_path/'report'
    import sys
    monkeypatch.setattr(sys,'argv',['replay_market.py','--predictions',str(p),'--odds-csv',str(q),
                                    '--output',str(out),'--offline'])
    rm.main()
    meta=json.loads((out/'manifest.json').read_text())
    assert meta['market_summary']['markets']['full']['graded_games']==1
    assert meta['market_summary']['markets']['f5']['pushes']==1
    assert 'Closing-price retrospective ROI' in (out/'backtest.html').read_text()
    assert json.loads((out/'run_status.json').read_text())['status']=='complete'
    # The source CSV remains an unmodified prediction, not a rewritten ledger.
    assert 'full_roi_status' not in pd.read_csv(p)


def test_archive_model_choice_is_not_substituted_for_replayed_choice():
    f,s=grade([row(xw_lean='BOS')],[odds(xw_lean='NYY',xw_full='L')])
    assert f.loc[0,'xw_lean']=='BOS'
    assert f.loc[0,'full_selection_ml']==-150
    assert f.loc[0,'full_profit_units']==pytest.approx(2/3)


# ESPN's response for CIN @ DET, 2025-06-15 (event 401695967), trimmed to the
# fields read: 2025 lists ESPN BET (58) and its live feed (59), no DraftKings.
ESPN_2025_ODDS={'items':[
    {'provider':{'id':'58','name':'ESPN BET'},
     'homeTeamOdds':{'open':{'moneyLine':{'american':'-175'}},'close':{'moneyLine':{'american':'-180'}}},
     'awayTeamOdds':{'open':{'moneyLine':{'american':'+145'}},'close':{'moneyLine':{'american':'+150'}}}},
    {'provider':{'id':'59','name':'ESPN BET - Live Odds'},
     'homeTeamOdds':{'open':{'moneyLine':{'american':'-175'}}},
     'awayTeamOdds':{'open':{'moneyLine':{'american':'+145'}}}}]}
ESPN_2025_F5={'pageCount':1,'items':[
    {'type':{'id':'136'},'team':{'$ref':'http://x/seasons/2025/teams/6?lang=en'},
     'odds':{'american':{'value':'-180','open':'-150'}}},
    {'type':{'id':'136'},'team':{'$ref':'http://x/seasons/2025/teams/17?lang=en'},
     'odds':{'american':{'value':'+140','open':'+120'}}},
    {'type':{'id':'137'},'team':{'$ref':'http://x/seasons/2025/teams/6?lang=en'},
     'odds':{'american':{'value':'+105'}}}]}


def test_espn_parsers_default_to_draftkings_and_read_espn_bet_only_when_asked(monkeypatch):
    urls=[]
    def get(url):
        urls.append(url)
        if url.endswith('/odds'):
            return ESPN_2025_ODDS
        return ESPN_2025_F5 if '/odds/58/' in url else {'items':[]}
    monkeypatch.setattr(rm.mb,'_get',get)
    # The production default is unchanged: a 2025 event has no DraftKings close.
    assert rm.mb._espn_close('401695967') is None
    assert rm.mb._espn_close('401695967',rm.mb.ESPN_BET)==dict(
        open_home_ml=-175,open_away_ml=145,close_home_ml=-180,close_away_ml=150)
    tid={'6':'home','17':'away'}
    assert rm.mb._espn_f5('401695967',tid)['f5_close_home_ml'] is None
    assert '/odds/100/propBets' in urls[-1]
    f5=rm.mb._espn_f5('401695967',tid,rm.mb.ESPN_BET)
    assert (f5['f5_close_away_ml'],f5['f5_close_home_ml'])==(140,-180)
    assert '/odds/58/propBets' in urls[-1]


def books(dk=None,bet=None,calls=None):
    def close(eid,provider=rm.mb.ESPN_DRAFTKINGS):
        (calls if calls is not None else []).append(('full',eid,provider))
        return {rm.mb.ESPN_DRAFTKINGS:dk,rm.mb.ESPN_BET:bet}[provider](eid)
    return close


def test_espn_bet_fills_only_games_without_a_draftkings_close_and_keeps_its_label(monkeypatch):
    calls=[]
    monkeypatch.setattr(rm.mb,'_statsapi_day',lambda d:[sched(),sched(2,gameDate='2026-04-01T23:00:00Z')])
    monkeypatch.setattr(rm.mb,'_espn_day',lambda d:{('NYY','BOS'):[event(),event('e2',start='2026-04-01T23:00:00Z')]})
    dk=lambda eid:dict(close_away_ml=120,close_home_ml=-150) if eid=='e1' else None
    bet=lambda eid:dict(close_away_ml=150,close_home_ml=-180)
    monkeypatch.setattr(rm.mb,'_espn_close',books(dk,bet,calls))
    monkeypatch.setattr(rm.mb,'_espn_f5',lambda eid,tid,provider=rm.mb.ESPN_DRAFTKINGS:None)
    rows=[row(),row(2)]
    f,s=rm.grade(pd.DataFrame(rows),fetch_missing=True,get=lambda *a:None)
    assert list(f.full_market_source)==['espn_draftkings_close','espn_espnbet_close']
    assert list(f.full_home_ml)==[-150,-180]
    # DraftKings is preferred: ESPN BET is never consulted for the DK game.
    assert ('full','e1',rm.mb.ESPN_BET) not in calls
    full=s['markets']['full']
    assert full['sources']=={'espn_draftkings_close':1,'espn_espnbet_close':1}
    by=full['by_source']
    assert by['espn_draftkings_close']['profit_units']==pytest.approx(100/150)
    assert by['espn_espnbet_close']['profit_units']==pytest.approx(100/180)
    assert sum(b['profit_units'] for b in by.values())==pytest.approx(full['profit_units'])
    assert by['espn_espnbet_close']['always_chalk_roi']==pytest.approx(100/180)
    assert s['markets']['f5']['exclusions']=={'market_unavailable':2}
    panel=rm.render_panel(f)
    assert 'espn_espnbet_close' in panel and 'ESPN BET is used only where' in panel


def test_no_book_with_a_two_sided_close_stays_unpriced(monkeypatch):
    monkeypatch.setattr(rm.mb,'_statsapi_day',lambda d:[sched()])
    monkeypatch.setattr(rm.mb,'_espn_day',lambda d:{('NYY','BOS'):[event()]})
    monkeypatch.setattr(rm.mb,'_espn_close',books(lambda e:None,lambda e:dict(close_away_ml=150,close_home_ml=None)))
    monkeypatch.setattr(rm.mb,'_espn_f5',lambda eid,tid,provider=rm.mb.ESPN_DRAFTKINGS:None)
    f,s=rm.grade(pd.DataFrame([row()]),fetch_missing=True,get=lambda *a:None)
    assert f.loc[0,'full_odds_status']=='market_unavailable'
    assert s['markets']['full']['graded_games']==0
    assert s['markets']['full']['by_source']=={}
