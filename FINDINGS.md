# Findings — verified, deliberately not acted on

Things found while aligning the docs to `woba+plat_consol_v4` that are real but
were **not** changed in that pass, because each needs a decision this repo makes
in a PR body rather than a doc commit. Each entry states what was verified and
what would settle it. Nothing here has been patched.

---

## 1. `WEIGHT_COL = "BBE"` is dead on the lineup path

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

**Why it wasn't deleted.** Removing the constant and the fallback branch is the
subtractive fix this repo's standards ask for, and by the argument above it is a
no-op. But "no-op" rests on reading, not on a run: if `batting_order` were ever
absent, behaviour would change from BBE-weighted to an equal mean, and that is
lean-path. Deleting it deserves a build against a frozen `SLATE_DATE` showing a
zero diff, in its own commit, rather than riding along with a docs change.

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

## 3. The xwOBA lens has no reliability gate

The platoon lens carries `low_sample`, `pit_low_sample` and `reliable`
(`:2831`, `:2877`), and `_pl_chip` greys the chip and prints a `prior-driven`
badge when they trip. **The primary lens has no equivalent** — there is no
`xw_reliable` anywhere.

The failure mode is worse than a visibly broken card. A starter absent from the
Savant leaderboard does not render `nan`: the `_shrink_one` call at `:2456`
returns his prior, so `pit_xwOBA` becomes a plausible-looking number, the lean
is computed almost entirely from the lineup half, and nothing marks it. Under
wOBA v4 this is *more* consequential, not less — at `K = 400` the prior already
supplies most of a published rate, so the gap between "we measured this arm" and
"we defaulted this arm" is invisible in the output value itself.

**What it needs.** A per-side boolean and a reason string, greyed chip in
`_xw_strip`, suppressed lean pill in `cmb_card`, and a ledger-filterable field.
Suppressing a lean is an abstention, so it changes what grades and bumps
`MODEL_TAG` — new record family, inherited units, argued in the PR. The v7
zero-as-abstention rule is the precedent for treating that as a first-class
outcome.

---

## 4. `Pos.` is the roster primary position, not the lineup card position

`gf_lineups` (`:757`) extracts ids only from the Savant `gf?game_pk=` payload.
`Pos.` comes from `load_people` → `primaryPosition.abbreviation` (`:583`) and is
rendered per lineup slot at `:3260`. A team fielding three shortstops on the
card is that: three players whose *roster* primary is SS, one of whom is playing
second tonight.

Display-only, so no `MODEL_TAG` implication. Two acceptable fixes: pull the
per-slot position out of the `gf` payload if it is reliably present (needs a
live fetch to confirm — not verified here), or relabel the column `Prim.` so it
stops asserting something false. Do not invent positions.

---

## 5. Smaller, verified, unfixed

- **Footer zone label.** `_legend_head` (`:4096`) renders
  `· first pitch times Pacific`, which names only the game clocks while the
  build stamp beside it is also Pacific (`_built_text_now`, `:5736`, uses
  `datetime.now(PT)`). The convention is deliberate and documented at
  `:3177-3185` — one zone, stated once — but the label undersells it. A
  one-word change, no behaviour.

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
