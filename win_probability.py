#!/usr/bin/env python3
"""Regularized home-win probabilities; chronological, probability-only audit.

Reads frozen xw_net and saved pregame markets. Never modifies the ledger,
leans, hybrid selections, or registered tests. See docs/win_probability.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "xwoba_home_logit_v1"
REQUIRED = (
    "game_pk", "game_date", "model_tag", "model_metric", "status", "xw_net",
    "full_home", "full_away", "snapshot_utc", "scheduled_start_utc", "lock_status",
)
PREDICTION_COLUMNS = (
    "game_pk", "game_date", "snapshot_utc", "xw_net", "home_won", "p_home",
    "p_prior_home", "pregame_p_home", "market_eligible", "market_reason",
    "pregame_market_utc", "train_before_date", "train_last_date", "train_n",
    "train_slates", "intercept", "slope_per_unit",
)


@dataclass(frozen=True)
class Settings:
    # Fixed engineering defaults, not selected by this evaluation's outcomes.
    min_train: int = 100
    min_slates: int = 7
    delta_unit: float = 0.020
    ridge: float = 1.0

    def __post_init__(self):
        if self.min_train < 2 or self.min_slates < 1:
            raise ValueError("Training needs at least two games and one slate")
        if not np.isfinite(self.delta_unit) or self.delta_unit <= 0:
            raise ValueError("delta_unit must be positive and finite")
        if not np.isfinite(self.ridge) or self.ridge <= 0:
            raise ValueError("ridge must be positive and finite")


@dataclass(frozen=True)
class LogitModel:
    intercept: float
    slope_per_unit: float
    delta_unit: float
    ridge: float

    def predict(self, delta):
        x = np.asarray(delta, dtype=float)
        if not np.isfinite(x).all():
            raise ValueError("Cannot predict from a missing or infinite delta")
        return np.exp(-np.logaddexp(0, -(self.intercept
                                       + self.slope_per_unit * x / self.delta_unit)))


def fit_logit(delta, outcome, settings=Settings()):
    """Minimize summed Bernoulli NLL + ridge/2 * (a*a + b*b).

Both coefficients are shrunk toward zero. Scaling is fixed, never learned
from future rows. The positive penalty also keeps one-class fits finite.
"""
    x, y = np.asarray(delta, float), np.asarray(outcome, float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 2:
        raise ValueError("Need matching one-dimensional arrays with >= 2 rows")
    if not np.isfinite(x).all() or not np.isin(y, [0., 1.]).all():
        raise ValueError("Need finite deltas and binary outcomes")
    design = np.column_stack([np.ones(len(x)), x / settings.delta_unit])
    beta = np.zeros(2)

    def objective(b):
        eta = design @ b
        return np.sum(np.logaddexp(0, eta) - y * eta) + settings.ridge * (b @ b) / 2

    for _ in range(100):
        p = np.exp(-np.logaddexp(0, -(design @ beta)))
        grad = design.T @ (p - y) + settings.ridge * beta
        if np.max(np.abs(grad)) < 1e-8:
            break
        hessian = design.T @ (design * (p * (1 - p))[:, None])
        step = np.linalg.solve(hessian + settings.ridge * np.eye(2), grad)
        scale, before = 1., objective(beta)
        while scale > 2 ** -30:
            candidate = beta - scale * step
            if objective(candidate) <= before - 1e-4 * scale * (grad @ step):
                break
            scale /= 2
        else:
            raise RuntimeError("Logistic fit failed its line search")
        beta = candidate
        if np.max(np.abs(scale * step)) < 1e-10:
            break
    else:
        raise RuntimeError("Logistic fit did not converge")
    return LogitModel(float(beta[0]), float(beta[1]), settings.delta_unit, settings.ridge)


def _utc(values):
    # Reject naive timestamps: silently assuming a timezone can admit a late row.
    text = values.astype("string")
    aware = text.str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", na=False)
    return pd.to_datetime(text.where(aware), utc=True, errors="coerce", format="mixed")


def as_family(model_tags):
    """Normalise one tag or many into the tuple `prepare_rows` filters on."""
    if isinstance(model_tags, str):
        model_tags = model_tags.split(",")
    return tuple(t.strip() for t in model_tags if str(t).strip())


def prepare_rows(ledger, model_tags):
    """One SCALE FAMILY of source versions, pregame features, final W/L outcomes.

A family and not a single tag, because this maps `xw_net` to a probability
and `xw_net`'s units are a property of `_SCALE_FAMILIES`, not of one
`MODEL_TAG`. Scoping to the running tag alone discards same-scale rows for
no reason and resets the sample at every bump -- which is how this module
went dark on 2026-09-18 and scored nothing for four days with only
`Warm-up/unscored` to say so. `RECORD_TAGS` is the wrong relation in the
other direction: it pools v12 with v13, which is a share of the WIN-LOSS
line across a deliberate scale change, and a delta calibrated across two
spreads is the mixture this repo already refuses elsewhere.

Zero is a valid delta here, even though the directional model abstains at
zero. Missing/infinite features are not zeros. Market availability never
determines eligibility for fitting the xwOBA-only mapping.
"""
    missing = sorted(set(REQUIRED) - set(ledger.columns))
    if missing:
        raise ValueError("Missing ledger columns: " + ", ".join(missing))
    family = as_family(model_tags)
    if not family:
        raise ValueError("No source model tag given")
    g = ledger.loc[ledger.model_tag.isin(family)].copy()
    audit = {"input_rows": len(ledger), "source_rows": len(g),
             "source_tags": list(family),
             "rows_by_tag": {str(k): int(v) for k, v
                             in g.model_tag.value_counts().items()},
             "excluded": {}}
    for c in ("xw_net", "full_home", "full_away", "game_pk"):
        g[c] = pd.to_numeric(g[c], errors="coerce")
    dates = pd.to_datetime(g.game_date, errors="coerce", format="%Y-%m-%d")
    snapshot, start = _utc(g.snapshot_utc), _utc(g.scheduled_start_utc)
    scores = g[["full_home", "full_away"]]
    checks = (
        ("not_xwoba", g.model_metric.eq("xwOBA")),
        ("not_graded", g.status.eq("graded")),
        ("invalid_identity_or_date", dates.notna() & np.isfinite(g.game_pk)
         & g.game_pk.gt(0) & g.game_pk.mod(1).eq(0)),
        ("missing_or_nonfinite_delta", np.isfinite(g.xw_net)),
        ("invalid_or_tied_score", np.isfinite(scores).all(axis=1)
         & scores.ge(0).all(axis=1) & scores.mod(1).eq(0).all(axis=1)
         & g.full_home.ne(g.full_away)),
        ("unverified_pregame_forecast", g.lock_status.eq("pregame")
         & snapshot.notna() & start.notna() & snapshot.lt(start)),
    )
    keep = pd.Series(True, index=g.index)
    for reason, valid in checks:
        valid = valid.fillna(False)
        audit["excluded"][reason] = int((keep & ~valid).sum())
        keep &= valid
    g = g.loc[keep].copy()
    g["game_date"] = dates.loc[keep].dt.strftime("%Y-%m-%d")
    g["snapshot_utc"] = snapshot.loc[keep]
    g["scheduled_start_utc"] = start.loc[keep]
    if g.duplicated(["game_date", "game_pk"]).any() or g.game_pk.duplicated().any():
        raise ValueError("Duplicate completed game identity; resolve before fitting")
    g["game_pk"] = g.game_pk.astype(int)
    g["home_won"] = g.full_home.gt(g.full_away).astype(int)
    q = pd.to_numeric(g.get("pregame_p_home", pd.Series(np.nan, index=g.index)),
                      errors="coerce")
    market_time = _utc(g.get("pregame_market_utc", pd.Series(index=g.index, dtype=str)))
    g["pregame_p_home"], g["pregame_market_utc"] = q, market_time
    g["market_reason"] = np.select(
        [~(np.isfinite(q) & q.gt(0) & q.lt(1)), market_time.isna(),
         market_time.gt(g.snapshot_utc)],
        ["missing_or_invalid_saved_probability", "missing_or_invalid_market_timestamp",
         "market_after_forecast"], default="eligible")
    g["market_eligible"] = g.market_reason.eq("eligible")
    audit["eligible_rows"] = len(g)
    audit["market_eligible_rows"] = int(g.market_eligible.sum())
    audit["market_exclusions"] = {str(k): int(v) for k, v in
                                  g.loc[~g.market_eligible, "market_reason"].value_counts().items()}
    return g.sort_values(["game_date", "game_pk"]).reset_index(drop=True), audit


def walk_forward(rows, settings=Settings()):
    """One fit per whole slate; only earlier dates can enter training.

If any forecast on the slate was saved on an earlier ET date, the training
cutoff moves back for the entire slate. No current-slate label enters a fit.
Historical outcome availability is date-based: the ledger lacks final-time
timestamps, so this replay is not a point-in-time certified forward trial.
"""
    parts, folds = [], []
    for date, cur in rows.groupby("game_date", sort=True):
        first_snapshot_date = cur.snapshot_utc.min().tz_convert("America/New_York").date().isoformat()
        cutoff = min(date, first_snapshot_date)
        train = rows.loc[rows.game_date.lt(cutoff)]
        n_slates = train.game_date.nunique()
        if len(train) < settings.min_train or n_slates < settings.min_slates:
            continue
        model = fit_logit(train.xw_net, train.home_won, settings)
        meta = dict(game_date=date, train_before_date=cutoff,
                    train_last_date=train.game_date.max(), train_n=len(train),
                    train_slates=n_slates, intercept=model.intercept,
                    slope_per_unit=model.slope_per_unit)
        folds.append(dict(**meta, test_n=len(cur), paired_market_n=int(cur.market_eligible.sum())))
        scored = cur.copy()
        scored["p_home"] = model.predict(cur.xw_net)
        # Jeffreys-smoothed earlier-slate home frequency, also out of sample.
        scored["p_prior_home"] = (train.home_won.sum() + .5) / (len(train) + 1)
        for k, v in meta.items():
            scored[k] = v
        parts.append(scored[list(PREDICTION_COLUMNS)])
    predictions = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=PREDICTION_COLUMNS)
    return predictions, pd.DataFrame(folds)


def _losses(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probability in scoring")
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return {"brier": (p - y) ** 2, "log_loss": -(y * np.log(p) + (1 - y) * np.log1p(-p))}


def _metrics(frame, column):
    if frame.empty:
        return {"n": 0}
    y, p = frame.home_won.to_numpy(float), frame[column].to_numpy(float)
    return dict(n=len(frame), slates=int(frame.game_date.nunique()),
                **{k: float(v.mean()) for k, v in _losses(y, p).items()},
                accuracy=float(((p >= .5) == y).mean()),
                mean_probability=float(p.mean()), actual_home_rate=float(y.mean()))


def _paired_interval(frame, loss_name, repeats, seed):
    if frame.empty:
        return {"difference": None, "ci95": None, "slates": 0}
    y = frame.home_won.to_numpy(float)
    delta = (_losses(y, frame.p_home)[loss_name]
             - _losses(y, frame.pregame_p_home)[loss_name])
    grouped = pd.DataFrame({"date": frame.game_date.to_numpy(), "delta": delta}).groupby("date").delta.agg(["sum", "count"])
    interval = None
    if len(grouped) >= 2 and repeats >= 2:
        rng = np.random.default_rng(seed)
        sums, counts = grouped["sum"].to_numpy(), grouped["count"].to_numpy()
        draws = rng.integers(0, len(grouped), (repeats, len(grouped)))
        means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        interval = np.quantile(means, [.025, .975]).tolist()
    return dict(difference=float(delta.mean()), ci95=interval, slates=len(grouped))


def evaluate(predictions, bootstrap_repeats=2000, seed=20260917):
    paired = predictions.loc[predictions.market_eligible.eq(True)]
    return {
        "all_oos": {"xwoba": _metrics(predictions, "p_home"),
                    "prior_home": _metrics(predictions, "p_prior_home")},
        "paired_market": {"xwoba": _metrics(paired, "p_home"),
                          "market": _metrics(paired, "pregame_p_home"),
                          "prior_home": _metrics(paired, "p_prior_home")},
        "model_minus_market": {name: _paired_interval(paired, name, bootstrap_repeats, seed)
                               for name in ("brier", "log_loss")},
        "uncertainty": "95% paired slate-bootstrap intervals, conditional on saved replay predictions; negative favors model. Not sequentially adjusted.",
        "bootstrap_repeats": bootstrap_repeats, "bootstrap_seed": seed,
    }


def calibration_rows(predictions):
    """Fixed probability bins on the identical paired OOS games."""
    paired = predictions.loc[predictions.market_eligible.eq(True)]
    result = []
    for name, column in (("xwoba", "p_home"), ("market", "pregame_p_home")):
        for lower, upper in zip([0, .2, .4, .6, .8], [.2, .4, .6, .8, 1.]):
            p = paired[column]
            g = paired.loc[p.ge(lower) & (p.le(upper) if upper == 1 else p.lt(upper))]
            result.append(dict(model=name, lower=lower, upper=upper, n=len(g),
                               mean_probability=float(g[column].mean()) if len(g) else None,
                               actual_home_rate=float(g.home_won.mean()) if len(g) else None))
    return pd.DataFrame(result)


def blind_reason(rows, predictions, settings, family):
    """Why this run scored nothing, or None when it scored something.

    Standing rule from CLAUDE.md, reached from the other side: when a function
    degrades silently by design, PRINT THE COUNT. A warm-up that swallows the
    whole sample rendered as `Warm-up/unscored: 47` beside two `no eligible
    games` lines -- true, and indistinguishable from an empty ledger or a
    broken join. This module went dark at the v13 bump on 2026-09-18 and
    nothing said so for four days.

    Derived from the settings and the rows, never a literal, so it names
    whichever threshold is actually binding and disappears on its own once
    either clears.
    """
    if len(predictions):
        return None
    slates = 0 if rows.empty else int(rows.game_date.nunique())
    short = []
    if len(rows) < settings.min_train:
        short.append(f"{settings.min_train - len(rows)} more eligible rows "
                     f"(has {len(rows)} of {settings.min_train})")
    if slates < settings.min_slates:
        short.append(f"{settings.min_slates - slates} more slates "
                     f"(has {slates} of {settings.min_slates})")
    need = "; ".join(short) if short else (
        "no slate cleared both thresholds with strictly earlier training rows")
    return ("SCORES NOTHING — this evaluation is BLIND on "
            + ", ".join(family) + f": needs {need}. "
            "A bump that starts a new scale family resets this sample, so the "
            "figures in docs/win_probability.md describe an earlier family "
            "and are not refreshed by this run.")


def report_text(summary):
    audit = summary["audit"]
    lines = ["xwOBA home-win probability — chronological retrospective evaluation",
             f"Source model: {summary['source_model_tag']}; calibrator: {VERSION}",
             "P(home win) = sigmoid(intercept + slope_per_unit * xw_net / delta_unit)",
             f"Settings: {summary['settings']}",
             f"Eligible: {audit['eligible_rows']} of {audit['source_rows']} source rows; exclusions: {audit['excluded']}",
             f"Saved pregame market eligible: {audit['market_eligible_rows']}; exclusions: {audit['market_exclusions']}",
             f"Warm-up/unscored: {summary['unscored_rows']}; OOS dates: {summary['oos_first_date']} to {summary['oos_last_date']}",
             "Market benchmark: pregame_p_home with market timestamp <= forecast snapshot < scheduled start. No closing fallback."]
    if summary.get("blind_reason"):
        lines.append(summary["blind_reason"])
    for group, label in (("all_oos", "All chronological OOS games"),
                         ("paired_market", "Same-game saved-market comparison")):
        lines.append("\n" + label + " (lower Brier/log loss is better)")
        for name, m in summary["metrics"][group].items():
            if m["n"]:
                lines.append(f"  {name:12s} n={m['n']:3d} slates={m['slates']:2d} Brier={m['brier']:.6f} log_loss={m['log_loss']:.6f} accuracy={m['accuracy']:.3f}")
            else:
                lines.append(f"  {name}: no eligible games")
    for name, m in summary["metrics"]["model_minus_market"].items():
        lines.append(f"  model - market {name}: {m['difference']}; 95% CI {m['ci95']}")
    lines.extend(["", summary["metrics"]["uncertainty"],
                  "Historical replay, not a registered prospective result. The design was specified after inspecting history.",
                  "Outcome availability is inferred from earlier slate dates; historical final-result timestamps are not stored.",
                  "The final-fit artifact is for later predictions only; it is never used to score its own training rows.",
                  "No directional rule, ledger row, model family, or registration is changed."])
    return "\n".join(lines) + "\n"


def main():
    import build_site

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=Path("data/mlb_lean_ledger.csv"))
    # The SCALE family, not the running tag: see `prepare_rows`. Comma-separated
    # so an operator can pin one tag or an older family by hand.
    parser.add_argument("--model-tag", default=",".join(build_site.SCALE_TAGS))
    parser.add_argument("--out-dir", type=Path, default=Path("win_probability_output"))
    args = parser.parse_args()
    settings = Settings()
    raw = args.ledger.read_bytes()
    rows, audit = prepare_rows(pd.read_csv(args.ledger, low_memory=False), args.model_tag)
    predictions, folds = walk_forward(rows, settings)
    summary = dict(source_model_tag=args.model_tag, calibrator_version=VERSION,
                   ledger_sha256=hashlib.sha256(raw).hexdigest(), settings=asdict(settings),
                   audit=audit, unscored_rows=len(rows) - len(predictions),
                   oos_first_date=None if predictions.empty else predictions.game_date.min(),
                   oos_last_date=None if predictions.empty else predictions.game_date.max(),
                   metrics=evaluate(predictions), final_fit=None,
                   blind_reason=blind_reason(rows, predictions, settings,
                                             audit["source_tags"]))
    if len(rows) >= settings.min_train and rows.game_date.nunique() >= settings.min_slates:
        summary["final_fit"] = dict(**asdict(fit_logit(rows.xw_net, rows.home_won, settings)),
                                   train_n=len(rows), train_first_date=rows.game_date.min(),
                                   train_last_date=rows.game_date.max(),
                                   use="Later-date predictions only; not scored in this report")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.out_dir / "predictions.csv", index=False)
    folds.to_csv(args.out_dir / "folds.csv", index=False)
    calibration_rows(predictions).to_csv(args.out_dir / "calibration.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = report_text(summary)
    (args.out_dir / "report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
