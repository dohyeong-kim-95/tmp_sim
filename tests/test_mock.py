import numpy as np
import pytest

from MOCKCalculator import ARRAY_SUM_KEY, MOCKCalculator, MockConfig, objective_key
from util._update_database import ARRAY_KEYS, LIST_KEYS, SCALAR_KEYS


@pytest.fixture
def mock(db_dir, code_to_ord):
    return MOCKCalculator(db_dir, MockConfig(k=3, trust_dist=0.05), code_to_ord).fit()


def test_stage1_builds_skeletons_without_arrays(db_dir, raw_dir, code_to_ord):
    """1단계는 catalog만 읽는다 — 배열 자리는 비어 있고 means만 채워진다."""
    mock = MOCKCalculator(db_dir, code_to_ord=code_to_ord)
    observed, rows_by_array_id = mock._read_catalog(db_dir)

    assert set(observed) == set(raw_dir["repeated_codes"]) | set(raw_dir["single_codes"])
    assert set(rows_by_array_id) == set(raw_dir["complete_array_ids"])
    for code in raw_dir["repeated_codes"]:
        obs = observed[code]
        assert obs.n_obs == 2                         # run_a + run_b
        assert all(obs.p[key] == [] for key in ARRAY_KEYS)
        assert set(obs.means) == set(LIST_KEYS) | set(SCALAR_KEYS)
        assert all(isinstance(v, float) for v in obs.means.values())


def test_stage2_fills_arrays_then_stage3_folds_them(db_dir, raw_dir, code_to_ord):
    """2단계가 슬라이스를 쌓고, 3단계가 그걸 원소별 True 비율로 접는다."""
    mock = MOCKCalculator(db_dir, code_to_ord=code_to_ord)
    observed, rows_by_array_id = mock._read_catalog(db_dir)

    mock._load_arrays(db_dir, observed, rows_by_array_id)
    for code in raw_dir["repeated_codes"]:
        for key in ARRAY_KEYS:
            slices = observed[code].p[key]
            assert len(slices) == observed[code].n_obs
            assert all(a.shape == raw_dir["shape"] and a.dtype == bool for a in slices)

    stacked = {code: np.stack(observed[code].p[ARRAY_KEYS[0]]) for code in observed}
    mock._build_p(observed)
    for code, obs in observed.items():
        assert obs.p[ARRAY_KEYS[0]].shape == raw_dir["shape"]
        assert np.array_equal(obs.p[ARRAY_KEYS[0]], stacked[code].mean(axis=0))


def test_fit_groups_repeated_observations(mock, raw_dir):
    for code in raw_dir["repeated_codes"]:
        assert mock.observed[code].n_obs == 2
    for code in raw_dir["single_codes"]:
        assert mock.observed[code].n_obs == 1
    assert len(mock.observed) == len(raw_dir["repeated_codes"]) + len(raw_dir["single_codes"])


def test_layer1_returns_consensus(mock, raw_dir):
    code = raw_dir["repeated_codes"][0]
    r = mock.predict(code)
    obs = mock.observed[code]

    assert r.layer == 1 and r.nearest_dist == 0.0 and not r.needs_test
    assert r.n_obs == obs.n_obs
    for key in ARRAY_KEYS:
        assert r.p_arrays[key].shape == raw_dir["shape"]
        assert r.p_arrays[key].min() >= 0.0 and r.p_arrays[key].max() <= 1.0
        assert np.array_equal(r.arrays[key], obs.p[key] >= 0.5)


def test_noise_shows_up_as_fractional_p(mock, raw_dir):
    # 반복 관측이 경계에서 흔들리므로 0/1이 아닌 원소가 존재해야 한다.
    fractional = 0
    for code in raw_dir["repeated_codes"]:
        for key in ARRAY_KEYS:
            p = mock.predict(code).p_arrays[key]
            fractional += int(((p > 0) & (p < 1)).sum())
    assert fractional > 0


def test_objectives_shape_and_meaning(mock, raw_dir):
    code = raw_dir["repeated_codes"][0]
    r = mock.predict(code)

    assert set(r.objectives) == {ARRAY_SUM_KEY} | {objective_key(k) for k in LIST_KEYS}
    assert r.objectives[ARRAY_SUM_KEY] == pytest.approx(
        sum(r.p_arrays[k].sum() for k in ARRAY_KEYS)
    )
    # list y의 목적값은 관측별 mean(list)의 평균이다.
    for key in LIST_KEYS:
        assert r.objectives[objective_key(key)] == pytest.approx(
            mock.observed[code].means[key]
        )
    # y5/y6는 batch 설정값이라 목적함수가 아니다.
    assert set(r.scalars) == set(SCALAR_KEYS)
    assert not set(SCALAR_KEYS) & set(r.objectives)


def test_result_has_no_combined_score(mock, raw_dir):
    # 스칼라화는 옵티마이저 책임 — MOCK은 단일 점수를 노출하지 않는다.
    r = mock.predict(raw_dir["repeated_codes"][0])
    assert not hasattr(r, "score")
    assert not hasattr(mock, "evaluate")


def test_layer2_for_unobserved_x(mock):
    near, far = "AAB", "ZZZ"
    assert near not in mock.observed and far not in mock.observed

    r_near, r_far = mock.predict(near), mock.predict(far)
    assert r_near.layer == 2 and r_far.layer == 2
    assert r_near.n_obs == 0
    assert r_near.nearest_dist < r_far.nearest_dist
    assert r_far.needs_test
    for key in ARRAY_KEYS:
        p = r_near.p_arrays[key]
        assert p.min() >= 0.0 and p.max() <= 1.0
        assert np.array_equal(r_near.arrays[key], p >= 0.5)


def test_layer1_never_needs_test(db_dir, code_to_ord):
    """관측된 x는 최근접 거리가 0이므로 trust_dist를 0으로 조여도 needs_test가 아니다."""
    for trust_dist in (0.0, 0.05):
        mock = MOCKCalculator(db_dir, MockConfig(k=3, trust_dist=trust_dist), code_to_ord).fit()
        for x in mock.observed:
            r = mock.predict(x)
            assert r.layer == 1
            assert r.nearest_dist == 0.0
            assert not r.needs_test


def test_layer2_weight_pulls_prediction_toward_nearest_neighbor(db_dir):
    """최근접 이웃을 인위적으로 아주 가깝게 두면 예측이 그 이웃 값으로 끌려가야 한다."""
    anchor, target = "AAA", "ZZZ"

    def code_to_ord_with_near_target(code):
        if code == target:                       # anchor 바로 옆에 놓는다
            return [ord(c) + 0.01 for c in anchor]
        return [ord(c) for c in code]

    mock = MOCKCalculator(
        db_dir, MockConfig(k=3, trust_dist=10.0), code_to_ord_with_near_target
    ).fit()
    assert target not in mock.observed

    pred = mock.predict(target)
    assert pred.layer == 2
    assert pred.nearest_dist < 0.01

    anchor_obs = mock.observed[anchor]
    # 배열은 anchor의 합의와 일치해야 한다.
    for key in ARRAY_KEYS:
        assert np.array_equal(pred.arrays[key], anchor_obs.p[key] >= 0.5)

    # list y 파생 목적값은 anchor 값에 붙는다. 관측값 전체 폭 대비 5% 미만 —
    # 거리 가중이 아니라 균등 평균이었다면 이 폭의 절반쯤 벗어난다.
    for key in LIST_KEYS:
        name = objective_key(key)
        values = [o.means[key] for o in mock.observed.values()]
        spread = max(values) - min(values)
        assert spread > 0

        anchor_value = anchor_obs.means[key]
        gap = abs(pred.objectives[name] - anchor_value)
        assert gap < 0.05 * spread
        assert gap < abs(pred.objectives[name] - np.mean(values))   # 균등 평균보다 가깝다
        for other_value in values:
            if abs(other_value - anchor_value) > 1e-9:
                assert gap < abs(pred.objectives[name] - other_value)


def test_trust_dist_controls_needs_test(db_dir, code_to_ord):
    strict = MOCKCalculator(db_dir, MockConfig(k=3, trust_dist=0.0), code_to_ord).fit()
    loose = MOCKCalculator(db_dir, MockConfig(k=3, trust_dist=10.0), code_to_ord).fit()
    assert strict.predict("ZZZ").needs_test
    assert not loose.predict("ZZZ").needs_test


def test_predict_requires_fit(db_dir, code_to_ord):
    with pytest.raises(RuntimeError, match="fit"):
        MOCKCalculator(db_dir, code_to_ord=code_to_ord).predict("AAA")
