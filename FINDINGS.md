# Findings — verified, with what was done about each

Things found while aligning the docs to `woba+plat_consol_v4`, none of which
were changed in that pass. Each entry states what was verified and what would
settle it; the heading says whether it has since been acted on.

**Status as of 2026-08-06.** Everything on this list is closed. §1 is deleted,
§3 is instrumented *and* abstaining (shipped as wOBA v5), §4 and the footer
label in §5 are fixed, and §2 settled itself.

---

## 1. `WEIGHT_COL = "BBE"` is dead on the lineup path — *deleted 2026-08-06*

`build_site.py:1808` defines `WEIGHT_COL = "BBE"`, and `aggregate_lineup:2360`
passes it to `lineup_weight(g, WEIGHT_COL)`. That looks like the lineup
composite is weighted by batted-ball events — a per-PA rate weighted by a
per-BBE denominator, which would systematically underweight three-true-outcome
bats. **It is not.** `lineup_weight` (`:2102`) returns slot-PA weights whenever
`USE_SLOT_PA_WEIGHTS` is on and `batting_order` is usable, and only falls back
to the passed column otherwise:

```python
if USE_SLOT_PA_WEIGHTS and BATTING_ORDER_COL in getattr(g, "columns", []):
    w = slot_pa_weights(g[BATTING_ORDER_COL])
    if w.notna().any():
        return w
return g.get(fallback_col)
```

`USE_SLOT_PA_WEIGHTS = True` (`:89`). `batting_order` is assigned
unconditionally as `enumerate(lu, start=1)` in `hitter_rows` (`:1342`) and
survives the column projection at `:1370`. `slot_pa_weights` maps 1–9 through
`LINEUP_SLOT_PA`, so `w.notna().any()` is never False for a lineup group. **The
fallback is unreachable for lineup aggregation.** This is the v4 (xwOBA
lineage) slot-PA change; the comment at `:2359` says so.

`BBE` itself is not dead — at `:1572` it correctly weights the batted-ball
baselines (`GB%`/`FB%`/`LD%`/`Pull%`…), which genuinely are per-BBE rates.

**Why it wasn't deleted at the time.** Removing the constant and the fallback
branch is the subtractive fix this repo's standards ask for, and by the argument
above it is a no-op. But "no-op" rested on reading, not on a run: if
`batting_order` were ever absent, behaviour would change from BBE-weighted to an
equal mean, and that is lean-path. It wanted a build against a frozen
`SLATE_DATE` showing a zero diff, in its own commit.

**How it was settled instead.** That build is not available — the environment's
network policy denies `statsapi.mlb.com` and `baseballsavant.mlb.com` at the
proxy (`CONNECT` → 403), so no live slate can be fetched here. The substitute is
the same measurement one step in from the fetch: frames built through the real
path (`build_tables` → `segment_pitcher_blocks` → the groupby
`aggregate_lineup` uses), with the weight vector `lineup_weight` returned
compared per group against the slot-PA vector alone. Six lineup groups, chosen
to be adversarial — a 10-man card, a hitter absent from the leaderboard, a side
with no BBE at all, and one with BBE stacked against the slot order — and the
returned vector was the slot vector on **every** group. Zero differences, so
every downstream number is identical by construction.

The same run shows it is not a vacuous check: had the fallback been reached,
those composites would have moved by up to **0.008 wOBA** (per-group deltas
+0.0012, +0.0077, −0.0082, −0.0064, +0.0001, +0.0004). Dead, not equivalent.

`lineup_weight` now takes only the group and returns `None` when the order is
unusable; `WEIGHT_COL` and the `set(cols) | {WEIGHT_COL}` coercion that existed
only to feed it are gone. The two tests that asserted the BBE fallback now
assert its absence — an equal mean, specifically *not* the .320 BBE answer —
and a new test drives the real frame builder and fails if any lineup group ever
stops carrying a usable batting order, which is the invariant that made the
branch unreachable. `BBE` stays in `STAT_COLS`: it is a column the leaderboard
supplies, not a branch, and it still weights the batted-ball anchors that are
genuinely per-BBE.

---

## 2. A version note asserted graded rows that did not exist — since settled

**Resolved 2026-08-06 by the rows arriving, not by an edit.** The four v3 rows
graded overnight, so the comment is now true. Recounted from
`data/mlb_lean_ledger.csv` on 2026-08-06: `woba+plat_consol_v3` is 4 graded, 0
pending, and `data/ledger_report.txt` prints
`wOBA v3 n=4  wOBA full 3-1 (0.750)  F5 4-0 (1.000)`. `woba+plat_consol_v4` is
still 11 pending, 0 graded. Nothing was patched; the entry stays as the record
of an assumption that happened to come true. The original finding follows.

The `_RECORD_FAMILIES` comment on `woba+plat_consol_v4` (`build_site.py:167`)
reads: *"The v3 family had its own graded rows and they stay immutable."*

Counted from `data/mlb_lean_ledger.csv` on 2026-08-05:

| tag | graded | pending |
|---|---:|---:|
| `woba+plat_consol_v2` | 15 | 0 |
| `woba+plat_consol_v3` | **0** | 4 |
| `woba+plat_consol_v4` | **0** | 11 |

wOBA v3 has never had a graded row. All 4 of its rows are from today's slate and
still pending. The claim was written the same day v3 shipped, describing rows
that did not exist yet — the third instance in this repo of a version note
asserting rows a build had not yet produced (see the `v8 has no rows` entry in
`CLAUDE.md`, and the `wOBA full 217-164` front-page incident).

**Why it wasn't corrected in the source.** Unlike v8's, this one is not settled.
Those 4 rows carry a pregame lock and may grade tonight under the v3 tag, at
which point the sentence becomes true by accident. A comment that is wrong today
and right tomorrow should not be rewritten twice. It is recorded in `CLAUDE.md`
under the latent-gap list instead, with the instruction to check the ledger
rather than the comment.

*That is what happened: the rows graded the next morning and the sentence is
true by outcome. The instruction stands unchanged — a version note is evidence
of what someone expected, not of what the ledger holds.*

---

## 3. The xwOBA lens has no reliability gate — *instrumented 2026-08-06; the abstention is still open*

**Correction first: this entry's opening sentence was wrong, and it was wrong in
the way this repo has a rule against.** It said `_pl_chip` greys the chip and
prints a `prior-driven` badge. There is no `_pl_chip` in `build_site.py`, and no
`_xw_strip` either — both were recalled, not read. What is real: the platoon
lens computes `low_sample`, `pit_low_sample` and `reliable` (`:2831`, `:2877`)
and passes `pl_reliable` into the card dict (`:3997`), where **nothing reads
it** — the platoon lens is no longer surfaced on the cards at all. So there was
no greyed-chip precedent to mirror; the primary lens was not missing a gate the
other lens had.

**What shipped.** `build_matchup` now records `starter_rate_basis`
(`measured` / `prior_only`) and `starter_rate_bf` per side, dumped, ledgered as
`sp_rate_basis_*` / `sp_rate_bf_*` (in `AUDIT_COLS` and `MODEL_FIELDS`, so a
pregame refresh updates them after a scratch), and rendered as a `prior only`
badge beside the starter. Incidence, measured over the 403 side-games in the
committed dumps: **6 (1.5%)** published a starter rate equal to the league prior
to full float precision, every one with a null `K%` — no leaderboard row. That
test only works pre-v4, when the prior was the league centre; under a personal
prior the defaulted value is no longer identifiable after the fact, which is the
argument for recording it at build time rather than inferring it later.

**The abstention shipped too, on 2026-08-06 as `woba+plat_consol_v5`** — asked
for directly rather than waiting on v4 incidence, which is worth recording since
this entry had argued for waiting. A side whose starter is `prior_only` now
publishes no lean, through the same NaN-edge path a missing bullpen already
takes; `starter_xwOBA` still records the prior, so the abstention's cause stays
auditable. New record family (the decided set changes), v4's scale family
(nothing is rescaled, and dropping the abstained games moves no cutoff) — both
halves measured, in the PR and in `CLAUDE.md`. The v7 zero-as-abstention rule
was the precedent; v5 is the first mechanism here that has actually produced a
graded-but-undecided row, since v7's never fires at full precision.

What the wait would have bought is still worth having: incidence under v4 is
unmeasured, and the 3.2%-of-games figure the decision was argued from comes from
the xwOBA lineage, where the prior was the league centre. Recheck it once v5 has
rows.

The original entry, minus its false premise:

The failure mode is worse than a visibly broken card. There is no `xw_reliable`
anywhere.

The failure mode is worse than a visibly broken card. A starter absent from the
Savant leaderboard does not render `nan`: the `_shrink_one` call at `:2456`
returns his prior, so `pit_xwOBA` becomes a plausible-looking number, the lean
is computed almost entirely from the lineup half, and nothing marks it. Under
wOBA v4 this is *more* consequential, not less — at `K = 400` the prior already
supplies most of a published rate, so the gap between "we measured this arm" and
"we defaulted this arm" is invisible in the output value itself.

---

## 4. `Pos.` is the roster primary position, not the lineup card position — *fixed 2026-08-06*

`gf_lineups` (`:757`) extracts ids only from the Savant `gf?game_pk=` payload.
`Pos.` comes from `load_people` → `primaryPosition.abbreviation` (`:583`) and is
rendered per lineup slot at `:3260`. A team fielding three shortstops on the
card is that: three players whose *roster* primary is SS, one of whom is playing
second tonight.

Display-only, so no `MODEL_TAG` implication. Two acceptable fixes were named:
pull the per-slot position out of the `gf` payload if it is reliably present
(needs a live fetch to confirm — still not verified), or relabel the column so
it stops asserting something false. **Neither, as written, was available: the
lineup table renders no header row at all** (`_lineup_details` says so in a
comment), so there was no `Pos.` label to rename — the false assertion is the
cell value, not a heading. Fixed where the claim actually lives: the cell now
carries `title='roster primary position, not tonight's lineup card'`, the same
pattern as the team-backfill asterisk one column over. Positions are still not
invented.

---

## 5. Smaller, verified, unfixed

- **Footer zone label — fixed 2026-08-06.** `_legend_head` rendered
  `· first pitch times Pacific`, which named only the game clocks while the
  build stamp beside it is also Pacific (`_built_text_now` uses
  `datetime.now(PT)`). The convention is deliberate and documented on
  `_fmt_pt_clock` — one zone, stated once — but the label undersold it. Now
  `· all times Pacific`. The test that guarded it pinned the old sentence
  verbatim and failed on the wording; it now asserts the claim (the zone is
  named, exactly once, and not scoped to one kind of clock), which is the
  frozen-literal-in-a-test anti-pattern from `CLAUDE.md` caught one more time.

- **`compare_v8_v9.py` compares against a version with no graded rows.** Already
  recorded in `CLAUDE.md` as inert and deliberately kept: the map is the
  authority on a historical question and deleting the entry would lose the
  answer. Repeated here only so it is not "rediscovered" and deleted as clutter
  — that is its own anti-pattern entry in the same file.

- **The pitch-mix noise budget is stale in both terms.** `MATCHUP_SITE.md`
  §"Why it is dark" computed a 69% noise-to-lean ratio at `K = 100` against a
  median `|xw_net|` of 0.0188. `K` is now 400 (cuts the numerator) and the
  scale family has reset twice (v3 compressed the denominator, v4 dispersed
  it). Flagged in place in that section rather than recomputed: the v4 pool is
  11 rows and its median `|xw_net|` of 0.0062 is noise at that size. Note the
  correction made 2026-08-06 — the section previously said the median was "not
  computable" because v4 has no graded rows, which reads the wrong rule.
  `lean_strength_scale()` counts pending rows too, since `|xw_net|` is a
  pregame quantity; what the ratio waits on is a bigger pool, not a graded one.
