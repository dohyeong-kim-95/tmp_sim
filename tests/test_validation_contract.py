"""검증 계약의 분기 — 소표본(erode/dilate), 대표본(교집합/합집합), 그 경계."""

import numpy as np
import pytest

from MOCKCalculator import (
    MOCKCalculator,
    MockConfig,
    array_bounds,
    dilate,
    erode,
    objective_key,
    scalar_bounds,
)
from util._update_database import ARRAY_KEYS, LIST_KEYS, SCALAR_KEYS


@pytest.fixture
def mock(db_dir, code_to_ord):
    return MOCKCalculator(db_dir, MockConfig(k=3, trust_dist=0.05), code_to_ord).fit()


def _small_sample_case(mock):
    """소표본 x와, 하한/상한 위반을 만들 여지가 있는 배열 키를 고른다."""
    for x in sorted(mock.observed):
        obs = mock.observed[x]
        if obs.n_obs >= mock.cfg.min_obs:
            continue
        for key in ARRAY_KEYS:
            lo, hi, _ = array_bounds(obs.stacks[key], mock.cfg)
            if lo.any() and not hi.all():
                return x, key
    pytest.fail("fixture에 소표본 x가 없다")


# --- 1. 소표본 분기 --------------------------------------------------------
def test_small_sample_uses_erode_dilate_rule(mock):
    x, key = _small_sample_case(mock)
    obs = mock.observed[x]
    assert obs.n_obs < mock.cfg.min_obs

    lo, hi, rule = array_bounds(obs.stacks[key], mock.cfg)
    assert rule == f"erode/dilate(d={mock.cfg.dilate_d})"
    assert np.array_equal(lo, erode(obs.consensus[key], mock.cfg.dilate_d))
    assert np.array_equal(hi, dilate(obs.consensus[key], mock.cfg.dilate_d))
    # 하한 ⊆ 합의 ⊆ 상한
    assert not (lo & ~obs.consensus[key]).any()
    assert not (obs.consensus[key] & ~hi).any()


def test_small_sample_rejects_prediction_below_eroded_lower_bound(mock):
    x, key = _small_sample_case(mock)
    lo, _, _ = array_bounds(mock.observed[x].stacks[key], mock.cfg)

    pred = mock.predict(x)
    missing = pred.arrays[key].copy()
    missing[np.unravel_index(np.flatnonzero(lo.ravel())[0], lo.shape)] = False

    pred.arrays[key] = missing
    report = mock.validate(x, pred)
    assert not report["ok"]
    assert report["arrays"][key]["below"] > 0
    assert report["arrays"][key]["rule"].startswith("erode/dilate")


def test_small_sample_rejects_prediction_above_dilated_upper_bound(mock):
    x, key = _small_sample_case(mock)
    _, hi, _ = array_bounds(mock.observed[x].stacks[key], mock.cfg)

    pred = mock.predict(x)
    extra = pred.arrays[key].copy()
    extra[np.unravel_index(np.flatnonzero(~hi.ravel())[0], hi.shape)] = True

    pred.arrays[key] = extra
    report = mock.validate(x, pred)
    assert not report["ok"]
    assert report["arrays"][key]["above"] > 0


def test_small_sample_scalar_uses_abs_tol(mock):
    x, _ = _small_sample_case(mock)
    cfg = mock.cfg
    obs = mock.observed[x]

    for values in [obs.scalars[SCALAR_KEYS[0]], obs.list_means[LIST_KEYS[0]]]:
        lo, hi, rule = scalar_bounds(values, cfg)
        assert rule == f"mean±{cfg.abs_tol}"
        assert hi - lo == pytest.approx(2 * cfg.abs_tol)

    # abs_tol 안쪽은 통과, 바깥은 거부.
    name = objective_key(LIST_KEYS[0])
    inside = mock.predict(x)
    inside.objectives[name] += cfg.abs_tol * 0.5
    assert mock.validate(x, inside)["ok"]

    outside = mock.predict(x)
    outside.objectives[name] += cfg.abs_tol * 1.5
    assert not mock.validate(x, outside)["objectives"][name]["ok"]


# --- 2. 대표본 분기과 그 경계 ---------------------------------------------
def _repeated_x(mock):
    for x in sorted(mock.observed):
        if mock.observed[x].n_obs == 2:
            return x
    pytest.fail("fixture에 반복 관측 x가 없다")


def test_envelope_rule_applies_exactly_at_min_obs(db_dir, code_to_ord):
    """n_obs == min_obs는 대표본이다 (off-by-one 방지)."""
    at_boundary = MOCKCalculator(db_dir, MockConfig(min_obs=2), code_to_ord).fit()
    above_boundary = MOCKCalculator(db_dir, MockConfig(min_obs=3), code_to_ord).fit()

    x = _repeated_x(at_boundary)
    assert at_boundary.observed[x].n_obs == 2

    stack = at_boundary.observed[x].stacks[ARRAY_KEYS[0]]
    lo, hi, rule = array_bounds(stack, at_boundary.cfg)
    assert rule == "intersection/union"
    assert np.array_equal(lo, stack.all(axis=0))
    assert np.array_equal(hi, stack.any(axis=0))

    # n_obs가 min_obs보다 하나 모자라면 즉시 소표본 규칙으로 갈린다.
    _, _, rule_below = array_bounds(stack, above_boundary.cfg)
    assert rule_below.startswith("erode/dilate")

    values = at_boundary.observed[x].scalars[SCALAR_KEYS[0]]
    assert scalar_bounds(values, at_boundary.cfg)[2] == f"mean±{at_boundary.cfg.z}σ"
    assert scalar_bounds(values, above_boundary.cfg)[2] == f"mean±{above_boundary.cfg.abs_tol}"


def test_envelope_rejects_prediction_outside_union(db_dir, code_to_ord):
    mock = MOCKCalculator(db_dir, MockConfig(min_obs=2), code_to_ord).fit()
    x = _repeated_x(mock)
    key = ARRAY_KEYS[0]
    stack = mock.observed[x].stacks[key]

    pred = mock.predict(x)
    assert mock.validate(x, pred)["ok"]            # 합의는 교집합~합집합 사이

    pred.arrays[key] = pred.arrays[key] | ~stack.any(axis=0)
    report = mock.validate(x, pred)
    assert not report["ok"]
    assert report["arrays"][key]["rule"] == "intersection/union"
    assert report["arrays"][key]["above"] > 0


def test_envelope_rejects_prediction_missing_intersection(db_dir, code_to_ord):
    mock = MOCKCalculator(db_dir, MockConfig(min_obs=2), code_to_ord).fit()
    x = _repeated_x(mock)
    key = ARRAY_KEYS[0]
    intersection = mock.observed[x].stacks[key].all(axis=0)
    assert intersection.any()

    pred = mock.predict(x)
    dropped = pred.arrays[key].copy()
    dropped[np.unravel_index(np.flatnonzero(intersection.ravel())[0], intersection.shape)] = False
    pred.arrays[key] = dropped

    report = mock.validate(x, pred)
    assert not report["ok"]
    assert report["arrays"][key]["below"] > 0
