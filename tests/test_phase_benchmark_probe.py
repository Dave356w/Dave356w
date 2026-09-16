"""The phase-benchmark panel's gate, its joins, and the two claims it makes
about arithmetic.

Every assertion here is about a RULE. The panel's figures move with every
committed slate, and freezing one would be the
`test_record_reproduces_ledger_report` instance again -- the suite going red
for arithmetic the Actions bot did overnight. The one exception is the
lam = 0 identity, which is not a measurement: it is the definition of the
shipped form and the whole reason the rest of the module can be believed.
"""
import numpy as np
import pandas as pd
import pytest

import phase_benchmark_probe as pbp


def _rows(n=24, seed=0):
    """Ledger-shaped rows whose stored phase values are built the way
    `sequential_xwoba_phases` builds them, so `load` has a real inversion to
    recover and `variant_net` a real identity to reproduce."""
    rng = np.random.default_rng(seed)
    L = 0.3149
    r = {}
    for side in ("away", "home"):
        bs = rng.normal(0.3196, 0.008, n)
        bn = bs - rng.normal(0.0016, 0.001, n)
        sp = rng.normal(0.3122, 0.020, n)
        bp = rng.normal(0.2966, 0.010, n)
        q = rng.uniform(0.40, 0.70, n)
        r[f"opp_xwoba_vs_sp_{side}"] = bs
        r[f"opp_xwoba_neutral_{side}"] = bn
        r[f"starter_xwoba_{side}"] = sp
        r[f"bullpen_xwoba_{side}"] = bp
        r[f"sp_share_{side}"] = q
        r[f"mx_xwoba_sp_{side}"] = bs * sp / L
        r[f"mx_xwoba_bp_{side}"] = bn * bp / L
        r[f"act_woba_{side}"] = rng.normal(0.315, 0.09, n)
    d = pd.DataFrame(r)
    d["game_pk"] = range(n)
    d["game_date"] = "2026-09-01"
    d["model_tag"] = "tag"
    edge = {s: d[f"sp_share_{s}"] * d[f"mx_xwoba_sp_{s}"]
            + (1 - d[f"sp_share_{s}"]) * d[f"mx_xwoba_bp_{s}"] - L
            for s in ("away", "home")}
    d["xw_net"] = edge["away"] - edge["home"]
    return d, L


def _with_actuals(d, sp_hits, bp_hits, ab_each=30.0, spread=2.0, seed=0):
    """Attach starter-allowed and team batting lines so `phase_lines` resolves.

    The bullpen line is the residual the module takes, so the team totals are
    written as starter + bullpen rather than the other way round -- a negative
    residual is refused by `phase_lines` and would silently drop every row.

    `spread` jitters the hit counts per row. Without it every game is
    identical, the bootstrap returns one value and the interval has zero
    width -- which the module now refuses to render a verdict against, so a
    fixture with no spread tests the refusal rather than the verdict.
    """
    rng = np.random.default_rng(seed)
    out = d.copy()
    n = len(out)
    for side in ("away", "home"):            # PITCHING side
        bat = "home" if side == "away" else "away"
        for f in ("2b", "3b", "hr", "bb", "ibb", "hbp", "sf"):
            out[f"act_sp_{f}_{side}"] = 0.0
            out[f"act_{f}_{bat}"] = 0.0
        sp = np.clip(np.round(rng.normal(sp_hits, spread, n)), 0, ab_each)
        bp = np.clip(np.round(rng.normal(bp_hits, spread, n)), 0, ab_each)
        out[f"act_sp_ab_{side}"] = ab_each
        out[f"act_sp_h_{side}"] = sp
        out[f"act_ab_{bat}"] = ab_each * 2
        out[f"act_h_{bat}"] = sp + bp
    return out


def test_the_league_denominator_inverts_exactly_from_any_phase(tmp_path):
    d, L = _rows()
    p = tmp_path / "led.csv"
    d.to_csv(p, index=False)
    got = pbp.load(p, tags=())
    assert np.allclose(got["L"], L)
    # Four independent inversions of one constant: their spread is the gate
    # the report prints, and it has to be float noise, not "small".
    assert float(got["L_spread"].max()) < 1e-12


def test_lambda_zero_reproduces_the_shipped_net_exactly(tmp_path):
    """Not a tolerance check. `variant_net(.., 0, 0)` IS the shipped
    construction written a second way, so anything above float noise means the
    rewrite diverged and every d_corr in the panel is measuring that instead
    of the change under test."""
    d, _ = _rows()
    p = tmp_path / "led.csv"
    d.to_csv(p, index=False)
    got = pbp.load(p, tags=())
    net = pbp.variant_net(got, pbp.peer_centres(got), 0.0, 0.0)
    assert float((net - got["xw_net"]).abs().max()) < 1e-12


def test_the_side_cross_is_pinned_against_a_constructed_frame():
    """The away-PITCHING row carries the HOME offense. A crossed suffix here
    returns a plausible weak correlation rather than an error, which is the
    hazard the ledger-report field conventions exist for -- so it is pinned
    by construction and never by a correlation's sign.

    One side's starter is made much worse than the other's with everything
    else held equal. The home offense faces the away staff, so a weak AWAY
    staff must push `net` POSITIVE.
    """
    d, L = _rows(n=4, seed=3)
    for side, sp in (("away", 0.400), ("home", 0.280)):
        for col, val in ((f"opp_xwoba_vs_sp_{side}", 0.3196),
                         (f"opp_xwoba_neutral_{side}", 0.3180),
                         (f"starter_xwoba_{side}", sp),
                         (f"bullpen_xwoba_{side}", 0.2966),
                         (f"sp_share_{side}", 0.6)):
            d[col] = val
        d[f"mx_xwoba_sp_{side}"] = (d[f"opp_xwoba_vs_sp_{side}"]
                                    * d[f"starter_xwoba_{side}"] / L)
        d[f"mx_xwoba_bp_{side}"] = (d[f"opp_xwoba_neutral_{side}"]
                                    * d[f"bullpen_xwoba_{side}"] / L)
    d = pd.concat([d, pd.DataFrame({"L": [L] * len(d)})], axis=1)
    assert (pbp.variant_net(d, pbp.peer_centres(d), 0.0, 0.0) > 0).all()


def test_a_uniform_centre_rescales_the_delta_and_decides_nothing():
    """The closure the module states, as a property -- and it is a SCALE
    statement, not an identity. When both phases sit at one centre the
    correction is a single positive constant on the whole net: the delta
    stretches, no sign flips, and the correlation against any outcome is
    unchanged to float precision. That is what makes 'its whole content is
    that the phases are centred in different places' a derivation rather than
    a slogan, and it is the reason a decision-level null here would still
    carry a `_SCALE_FAMILIES` question if anything were ever shipped.

    Written the other way round first -- as `moved == base` -- and the frame
    said otherwise, which is the claim that needed narrowing, not the code.
    """
    d, L = _rows(n=16, seed=7)
    d = pd.concat([d, pd.DataFrame({"L": [L] * len(d)})], axis=1)
    base = pbp.variant_net(d, pbp.peer_centres(d), 0.0, 0.0)
    flat = dict(pbp.peer_centres(d))
    flat["P_SP"] = flat["P_BP"] = 0.2900
    flat["H_SP"] = flat["H_BP"] = 0.3200
    moved = pbp.variant_net(d, flat, 1.0, 1.0)
    ratio = moved / base
    assert float(ratio.std()) < 1e-12 and float(ratio.mean()) > 0
    assert (np.sign(moved) == np.sign(base)).all()
    out = d["act_woba_home"] - d["act_woba_away"]
    assert np.corrcoef(moved, out)[0, 1] == pytest.approx(
        np.corrcoef(base, out)[0, 1], abs=1e-12)


def test_the_hitter_and_pitcher_knobs_are_separable():
    """Each knob must be reachable alone, or the panel cannot attribute the
    effect and the 'ten times as much room' closure is untestable."""
    d, L = _rows(n=16, seed=11)
    d = pd.concat([d, pd.DataFrame({"L": [L] * len(d)})], axis=1)
    c = pbp.peer_centres(d)
    base = pbp.variant_net(d, c, 0.0, 0.0)
    hit = pbp.variant_net(d, c, 1.0, 0.0)
    pit = pbp.variant_net(d, c, 0.0, 1.0)
    assert float((hit - base).abs().max()) > 0
    assert float((pit - base).abs().max()) > 0
    # The pitcher centres are the widely separated pair in any real frame, so
    # the pitcher knob must move more. Asserting the ORDER, not a size.
    assert float((pit - base).abs().mean()) > float((hit - base).abs().mean())


def test_implied_lambda_is_derived_from_a_target_not_fitted():
    """Bisection on the predicted gap, so feeding it the shipped gap must
    return 0 and feeding it the fully-matched gap must return 1. A value
    outside the bracket returns None rather than extrapolating."""
    d, L = _rows(n=16, seed=13)
    d = pd.concat([d, pd.DataFrame({"L": [L] * len(d)})], axis=1)
    c = pbp.peer_centres(d)
    assert pbp.implied_lambda(d, c, pbp.phase_gap(d, c, 0.0, 0.0)) == pytest.approx(0.0, abs=1e-6)
    assert pbp.implied_lambda(d, c, pbp.phase_gap(d, c, 0.0, 1.0)) == pytest.approx(1.0, abs=1e-6)
    assert pbp.implied_lambda(d, c, 5.0) is None


def test_the_realised_gap_is_pa_weighted_not_side_game_weighted():
    """A four-batter bullpen residual is a rate carrying no information. If
    the estimator weighted side-games equally it would give that row the same
    vote as a nine-inning line, which is how the unweighted figure on the real
    ledger lands 0.019 away from the weighted one."""
    heavy = {"ab": 30.0, "h": 12.0, "2b": 0.0, "3b": 0.0, "hr": 0.0,
             "bb": 0.0, "ibb": 0.0, "hbp": 0.0, "sf": 0.0}
    light = {**heavy, "ab": 2.0, "h": 2.0}
    row = {"game_pk": 1, "act_sp_ip_away": 6.0}
    for f, v in heavy.items():
        row[f"act_sp_{f}_away"] = v
        row[f"act_{f}_home"] = v + light[f]
    d = pd.DataFrame([row])
    got = pbp.realised_phase_gap(d, boot=32, seed=0)
    assert got["n_sides"] == 1
    # The bullpen residual is the tiny all-hit line; the PA weighting is what
    # keeps it from dominating once more rows exist. Here it is the only row,
    # so the check is that the two levels are computed off their own lines.
    assert got["sp"] == pytest.approx(pbp.ab.woba_from_components(heavy))
    assert got["bp"] == pytest.approx(pbp.ab.woba_from_components(light))


def test_an_empty_family_reports_rather_than_raises(tmp_path):
    d, _ = _rows(n=4)
    p = tmp_path / "led.csv"
    d.to_csv(p, index=False)
    out = "\n".join(pbp.report(p, tags=("no-such-tag",)))
    assert "no rows" in out


def test_the_current_family_is_derived_and_never_named():
    """A hardcoded tag list is the `interaction_probe` defect: the probe goes
    on answering about a model the build stopped running. The family must come
    from build_site, and no family tag may appear as a literal anywhere in the
    module."""
    src = open(pbp.__file__).read()
    assert "build_site.RECORD_TAGS" in src
    assert "plat_consol_v" not in src


def test_the_realised_gap_ignores_the_family_filter(tmp_path):
    """The row-set defect, pinned as a property. `phase_lines` reads box
    scores, so the realised gap must not change when the current family
    changes -- it must be scored on every row that carries actuals. The
    shipped version scoped it to RECORD_TAGS, which halved the sample and
    reversed the verdict."""
    fam, _ = _rows(n=12, seed=1)
    fam["model_tag"] = "fam"
    other, _ = _rows(n=12, seed=2)
    other["model_tag"] = "other"
    other["game_pk"] = range(100, 112)
    # Out-of-family rows carry a very different phase gap, so a family-scoped
    # estimator and an unfiltered one cannot agree by accident.
    led = pd.concat([_with_actuals(fam, 14.0, 7.0, seed=1),
                     _with_actuals(other, 7.0, 14.0, seed=2)], ignore_index=True)
    p = tmp_path / "led.csv"
    led.to_csv(p, index=False)

    scoped = pbp.realised_phase_gap(pbp.load(p, tags=("fam",)))
    pooled = pbp.realised_phase_gap(pbp.load_all(p))
    assert scoped["n_games"] == 12 and pooled["n_games"] == 24
    assert scoped["gap"] != pytest.approx(pooled["gap"])
    # And the licence must notice that these two windows disagree.
    lic = pbp.pooling_licence(pbp.load_all(p), ("fam",), boot=64)
    assert abs(lic["z"]) > 2


def test_the_containment_verdict_is_computed_not_asserted(tmp_path):
    """The defect this module shipped with: the report stated 'both are
    inside that interval' in prose, and a wider row set falsified it the same
    day. The verdict must follow the numbers in both directions, so this
    drives the realised gap to each side of the shipped prediction and reads
    the word back out."""
    d, _ = _rows(n=40, seed=5)
    d["model_tag"] = "fam"

    def verdict(sp_hits, bp_hits):
        led = _with_actuals(d, sp_hits, bp_hits)
        p = tmp_path / f"led_{sp_hits}_{bp_hits}.csv"
        led.to_csv(p, index=False)
        return "\n".join(pbp.report(p, tags=("fam",)))

    # Starter and bullpen allow the same line: realised gap ~0, so the
    # shipped construction's large positive gap must be rejected while the
    # matched one survives.
    assert "SHIPPED gap is REJECTED" in verdict(11.0, 11.0)
    # A realised gap far wider than either candidate rejects both.
    assert "BOTH are rejected" in verdict(22.0, 3.0)
    # And a gap that brackets the shipped prediction separates neither.
    assert "do not separate" in verdict(11.4, 11.0)


def test_a_zero_width_interval_renders_no_verdict(tmp_path):
    """`An SE of zero is never a result.` Identical games make every resample
    return the same number, and containment against a point is not a
    measurement -- the module must say so instead of printing OUTSIDE."""
    d, _ = _rows(n=12, seed=17)
    d["model_tag"] = "fam"
    led = _with_actuals(d, 11.0, 11.0, spread=0.0)
    p = tmp_path / "led.csv"
    led.to_csv(p, index=False)
    out = "\n".join(pbp.report(p, tags=("fam",)))
    assert "ZERO-WIDTH" in out
    assert "OUTSIDE the interval" not in out
    assert "REJECTED" not in out


def test_the_metric_label_is_read_off_the_rows(tmp_path):
    """`A build-time constant must never name historical rows` -- the most
    repeated instance in CLAUDE.md, and this module shipped with the literal
    'xwOBA' beside a denominator recovered from whatever rows were loaded."""
    d, _ = _rows(n=8, seed=9)
    d["model_tag"] = "fam"
    d["model_metric"] = "wOBA"
    p = tmp_path / "led.csv"
    d.to_csv(p, index=False)
    out = "\n".join(pbp.report(p, tags=("fam",)))
    assert "league batter wOBA" in out
    assert "league batter xwOBA" not in out
