# F5 (1st-5-innings) market capture — validation & first results

> **Historical document (2026-07-21 data, v5-era model; current model is v10).**
> The market-capture mechanics, the data-quality audit, and the mapping fit are
> unchanged and still describe today's code. Two things have moved:
>
> - **The "v6 promotion gate" below was never taken up.** At the time, v6 was
>   planned as a unified vs-hand-xwOBA lens, and this document pre-registered a
>   gate for it: beat `ops_net`'s r = 0.26 against ε on held-out games. v6
>   shipped as something else entirely — the expected-IP starter/bullpen
>   blend — and the unified vs-hand-xwOBA lens has still not been built. The
>   gate stands unclaimed for whatever eventually implements that idea; it is
>   *not* a gate v6 passed.
> - **The platoon lens left the cards.** It is still computed and graded into
>   the ledger for auditing, but the display is xwOBA-only, so the "load-bearing"
>   framing below describes an analytical finding, not what the site shows.
>
> The n=203 figures are as of 2026-07-21. The ledger now holds 310 graded rows;
> see `docs/build_logic_validation.md` for the refreshed vs-market read.

**What this adds.** DK "1st 5 Innings Moneyline" open/close per ledger row
(`f5_open_*_ml`, `f5_close_*_ml`, `f5_close_p_home`), extracted from the same
ESPN event already joined and score-verified by `attach_market`. The platoon
lean is now graded against the market it actually targets; previously it was
graded outcome-only (and, in `vs_market_summary`, against the *full-game*
close — a horizon mismatch).

## Data-quality audit (2026-07-07..07-20, 154 games)

The F5 lines live in the event's `propBets` child (type.id=136), which has no
`close` block — only `value` (last posted price) and `open`. Whether the
post-final `value` is the pregame close was validated before wiring:

- **Stamp semantics.** `lastUpdated` marks the last item *touch*, including
  settlement bookkeeping — ~31% of ML items stamp post-start. Stamps must not
  be read as price-change times.
- **Movement by stamp class (decisive).** |implied(value) − implied(open)|:
  pregame-stamped p99 = 5.9 pts, post-start-stamped p99 = 5.9 pts, 0/302
  sides > 10 pts. Live prices frozen mid-F5 would show 20–40 pt moves.
- **Lookahead detector.** Brier of the F5 close on the realized F5 winner vs
  a probit-mapped full-game baseline: gain −0.0005 (n=131). A contaminated
  price would slash Brier; a legitimate close cannot.
- **Vig.** F5 overround ~6.2% vs full-game ~4.6% — normal derivative premium.
- **Mechanical mapping null.** probit(F5 cond) vs probit(FG) has a Poisson-
  derived null slope of ~0.905 at MLB totals (tie-conditioning re-inflates
  the F5 favorite; extras-as-coinflip compresses the FG favorite). The naive
  sqrt(5/9)=0.745 ignores both effects and is wrong.

**Retention erodes rather than cliffs**: full at ~2 weeks, 5/15 games at ~3
months (items pruned from surviving prop sets), zero for the prior season.
The daily pass is what stops the bleeding.

## Ledger backfill (2026-07-02..07-21)

203/204 graded rows enriched; the single skip is the All-Star Game
(`statsapi join ambiguous`), correctly rejected by score verification.
Pending rows never receive F5 closes (`run_market_update.py` invariant).

## First results (n=203 unless noted)

- Mapping fit: slope **0.934 (se 0.018)** vs null 0.905 → modest composition
  loading (FG favorites tend to own the better starter). Residual sd
  **0.049 probit ≈ 0.020 prob at p=.5** — the market's game-specific SP/BP
  composition variance, the quantity model signals are tested against.
- Platoon lean vs F5 close: **74-58 (20 push), z +0.47, −1.55u** — first
  correctly-benchmarked record (vs z −0.14 at the mismatched full-game
  horizon). Noise-range, but the estimand finally matches.
- **ε ~ model signal** (ε = market F5 residual off the fitted FG map):
  - `ops_net` (platoon lens): **r = +0.263, t = +3.34, n = 152** —
    ~+0.5 prob-pts of F5 richness per 1 sd of signal.
  - `xw_net` (xwOBA lens): r = +0.137, t = +1.96, n = 203 — borderline.
  - corr(xw_net, ops_net) = 0.72: these are correlated looks, not two
    independent confirmations; the platoon result is the load-bearing one.

This is *validation, not edge*: agreeing with the market's composition
estimate shows the signal is real; beating the close remains the profit
standard. Notably the handedness-split lens — not the season-aggregate xwOBA
lens — carries the F5-composition signal, which sharpens the case for a unified
vs-hand-xwOBA lens and gives it a pre-registrable promotion gate: **its net
must beat `ops_net`'s r = 0.26 against ε on held-out games.**

> **Status.** That lens was pencilled in as "v6" when this was written. It was
> never built — v6 became the expected-IP starter/bullpen blend. The gate above
> remains open and applies to whatever implements the vs-hand-xwOBA idea, which
> as of v10 is still nothing. The nearest live relative is the pitch-mix shadow
> arm (`docs/pitch_mix_theory.md`), which is a different decomposition and
> carries its own separate gate.

## Standing caveats

- A price that never re-stamped could hide an unposted move; one week of
  parallel near-close pregame captures alongside CI remains the cheap final
  confirmation that untouched also means unmoved.
- ε is refit each analysis; store closes (facts), not residuals.
