"""Guards on the SP-vs-lineup weight fit's symmetry test.

The fit is diagnostic output -- nothing it produces feeds back into a lean, a
delta or a grade -- so its failure mode is not a wrong prediction. It is a
printed number that reads as a measurement when the data cannot support one,
which is the shape CLAUDE.md files under "public claims the data can't
support".
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import grade_leans as gl


def _fit(n=400, seed=0, b_lineup=0.0, b_sp=0.0, rho=0.0):
    """A logit fit on correlated inputs, returning (b, se, cov)."""
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rho * z1 + np.sqrt(max(1 - rho**2, 0.0)) * rng.normal(size=n)
    X = np.column_stack([np.ones(n), z1, z2])
    p = 1.0 / (1.0 + np.exp(-(b_lineup * z1 + b_sp * z2)))
    y = (rng.uniform(size=n) < p).astype(float)
    return gl._logit_fit(X, y)


def test_logit_fit_returns_the_full_covariance():
    """The contrast's variance needs the off-diagonal term, so the diagonal
    alone is not enough -- returning only `se` is what forced the ratio form."""
    b, se, cov = _fit()
    assert cov.shape == (3, 3)
    assert se == pytest.approx(np.sqrt(np.diag(cov)))
    assert cov == pytest.approx(cov.T)          # symmetric


def test_the_contrast_is_zero_when_native_weights_are_equal():
    """`w = 1` is `b_sp/sd_sp == b_lineup/sd_lu`. Build exactly that and the
    contrast must vanish, whatever the input scales are."""
    b = np.array([0.0, 0.40, 0.20])             # per-sd coefficients
    cov = np.eye(3) * 0.01
    # sd_lu / sd_sp = 2 makes the native weights equal: 0.40/1 vs 0.20/0.5.
    diff, se, z = gl.symmetry_contrast(b, cov, sd_lineup=1.0, sd_sp=0.5)
    assert diff == pytest.approx(0.0, abs=1e-12)
    assert z == pytest.approx(0.0, abs=1e-12)
    assert se > 0


def test_the_contrast_uses_the_covariance_not_just_the_diagonal():
    """THE REASON `_logit_fit` GAINED A RETURN VALUE.

    Two positively correlated coefficients have a SMALLER difference variance
    than the naive sum of their variances. Ignoring the off-diagonal term would
    overstate the se and hide a real departure.
    """
    b = np.array([0.0, 0.5, 0.1])
    v = 0.04
    naive = np.sqrt(v + v)                       # what diag-only would give
    pos = np.array([[v, 0, 0], [0, v, 0.03], [0, 0.03, v]])
    neg = np.array([[v, 0, 0], [0, v, -0.03], [0, -0.03, v]])
    _d, se_pos, _z = gl.symmetry_contrast(b, pos, 1.0, 1.0)
    _d, se_neg, _z = gl.symmetry_contrast(b, neg, 1.0, 1.0)
    assert se_pos < naive < se_neg
    assert se_pos == pytest.approx(np.sqrt(v + v - 2 * 0.03))
    assert se_neg == pytest.approx(np.sqrt(v + v + 2 * 0.03))


def test_a_real_departure_is_detected():
    """The test must be able to fire, or it is decoration."""
    b, se, cov = _fit(n=4000, seed=3, b_lineup=0.9, b_sp=0.0)
    _diff, _se, z = gl.symmetry_contrast(b, cov, 1.0, 1.0)
    assert z is not None and abs(z) > 2


def test_equal_weight_is_not_flagged():
    b, se, cov = _fit(n=4000, seed=4, b_lineup=0.4, b_sp=0.4)
    _diff, _se, z = gl.symmetry_contrast(b, cov, 1.0, 1.0)
    assert z is not None and abs(z) < 2


def test_degenerate_inputs_return_no_z_rather_than_a_fabricated_one():
    b = np.array([0.0, 0.5, 0.1])
    cov = np.eye(3) * 0.01
    for kwargs in (dict(sd_lineup=1.0, sd_sp=0.0),
                   dict(sd_lineup=np.nan, sd_sp=1.0),
                   dict(sd_lineup=1.0, sd_sp=np.nan)):
        assert gl.symmetry_contrast(b, cov, **kwargs)[2] is None
    # A zero-variance contrast yields no z either -- never a divide by zero.
    assert gl.symmetry_contrast(b, np.zeros((3, 3)), 1.0, 1.0)[2] is None


def test_the_ratio_form_is_gone():
    """It was `implied w = b_sp/b_lineup`, printed without a standard error
    because it has none. Kept as a named regression: the argument for dropping
    it lives in symmetry_contrast's docstring, so prose mentioning it is fine
    and a live division is not."""
    import ast
    tree = ast.parse(open(gl.__file__, encoding="utf-8").read())
    # Blank every docstring, then unparse: comments are already gone and this
    # removes prose. `symmetry_contrast`'s docstring names the retired form on
    # purpose -- that is the record of why it went -- so a substring search
    # over the raw text cannot tell the argument apart from the artefact.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body[0].value.value = ""
    code = ast.unparse(tree)
    assert "raw_sp / raw_lu" not in code
    assert "implied w" not in code
    assert "symmetry_contrast" in code


def test_the_gate_is_a_numerical_floor_not_an_evidence_threshold():
    """It was 120, sized to suppress the unreadable ratio. Coefficients printed
    with standard errors are honest at any n, so the floor now only has to keep
    the logit from returning a meaningless covariance."""
    assert gl.N_FIT_MIN <= 60
