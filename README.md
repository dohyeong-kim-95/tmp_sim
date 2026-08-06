# tmp_sim

비싸고 noisy한 실험 평가기(이하 **TRUE_CALCULATOR**)를 대상으로 하는 최적화 작업의
보조 부품 두 개를 담는 저장소.

## 1. 배경

- 입력 `X`는 string code. `optimizer.py`의 `code_to_ord` / `ord_to_code`로
  ordinal 정수 리스트와 상호 변환된다.
- TRUE_CALCULATOR의 출력은 batch 단위로:
  - bool 5D array 2개 (내용은 볼록한 blob 형태)
  - list 2개
  - scalar 2개 (batch 공통 설정값)
- 최종 목표는 **bool array의 합이 크고 scalar는 작은 X**를 GA/SA 계열로 찾는 것.
- 같은 X라도 실행마다 출력이 조금씩 다르다(noise). 따라서 한 X에 대한 반복 관측을
  모으는 것 자체가 데이터 자산이다.

TRUE_CALCULATOR 호출은 비싸므로, 축적된 실행 로그로 TRUE를 흉내내는 서로게이트를
두고 후보 대부분을 서로게이트로 걸러낸다. 이 저장소는 그 두 부품을 담는다.

- **A. ingest 파이프라인** (`util/ingest.py`): `raw/*.jsonl` → `database/`
- **B. 서로게이트** (`MOCKCalculator.py`): `database/`로 TRUE를 흉내냄

`optimizer.py`(GA/SA 본체, `code_to_ord`/`ord_to_code`)와 TRUE_CALCULATOR 자체는
이 저장소 밖에 있으며, 여기서는 **import만 한다**.

## 2. 구조

```
README.md
.gitignore
util/
  ingest.py          # raw/*.jsonl -> database/ (A)
MOCKCalculator.py    # 서로게이트 (B)

raw/                 # (gitignore) TRUE_CALCULATOR 실행 로그, run 하나당 파일 하나
database/            # (gitignore) ingest 산출물
  catalog.jsonl      #   x 하나당 한 줄
  arrays/{array_id}.npz
```

`raw/`, `database/`는 데이터라서 버전 관리하지 않는다. `database/`는 언제든
`raw/`로부터 재생성 가능한 파생물이다.

## 3. 플레이스홀더

실제 실험의 키 이름·차원·수치는 이 저장소에 반영되어 있지 않다. 아래 상수는
전부 **플레이스홀더**이며 실제 로그에 맞춰 고쳐야 한다.

| 위치 | 상수 | 의미 |
| --- | --- | --- |
| `util/ingest.py` | `X_KEY` | X(코드 문자열 batch)의 키 이름 |
| `util/ingest.py` | `ARRAY_KEYS` | bool 5D array 2개의 키 이름 |
| `util/ingest.py` | `LIST_KEYS` | list y 2개의 키 이름 |
| `util/ingest.py` | `SCALAR_KEYS` | scalar y 2개의 키 이름 |
| `util/ingest.py` | `AXIS1_SIZE` | 배열 내부 축1의 고정 크기 (축 뒤바뀜 판정용) |
| `MOCKCalculator.py` | `MockConfig` | k, trust_dist, min_obs, dilate_d, z, abs_tol |

## 4. A. ingest 파이프라인

### 입력 포맷 (`raw/*.jsonl`)

한 줄 = `{"<iteration>": {"<키>": <값>}}`. 한 iteration의 정보가 **두 줄**
(배열 줄 + 스칼라 줄)에 나뉘어 기록되므로, 같은 iteration의 줄들을 **null이 아닌
값으로 병합**해서 하나의 레코드로 만든다.

```jsonl
{"1": {"x": ["A", "B"], "y1_key": [[...]], "y3_key": [[...]], "y5_key": null}}
{"1": {"x": null,       "y1_key": null,    "y3_key": null,    "y5_key": 0.5}}
```

run(파일)마다 iteration 번호가 1부터 리셋되므로 고유 키는 **(파일명, iteration)**.
`array_id = "{파일 stem}:{iteration}"`으로 명명한다.

### batch 정렬 규약

`X`와 배열/리스트 y의 첫 축은 batch로 정렬되어 있다. 즉 `X[i]`의 결과가 `y[i]`.
스칼라 y는 batch 공통 설정값이므로 batch 내 모든 x가 같은 값을 갖는다.

### 축 뒤바뀜 정규화

일부 run은 배열 **내부 축 0과 1이 뒤바뀌어** 기록된다. 내부 축1이 고정 크기
(`AXIS1_SIZE`)를 갖는 점을 이용해 판정한다. batch 축을 포함한 기록 배열은 6D
`(batch, i0, i1, i2, i3, i4)`이고,

- `shape[2] == AXIS1_SIZE` 이고 `shape[1] != AXIS1_SIZE` → 정상
- `shape[1] == AXIS1_SIZE` 이고 `shape[2] != AXIS1_SIZE` → 뒤바뀜, `swapaxes(1, 2)`
- 그 외(둘 다 같거나 둘 다 다름) → **판정 불가, 에러로 중단**

조용히 통과시키면 데이터베이스가 오염되므로 중단이 맞다.

### 출력 1: `database/catalog.jsonl`

batch를 풀어 **x 하나당 한 줄**.

```json
{"x": "<code>", "array_id": "run_a:1", "batch_pos": 0,
 "y3_key": [...], "y4_key": [...], "y5_key": 0.5, "y6_key": 1.2}
```

같은 x가 여러 번 관측되면 줄이 여러 개 생긴다. 이것이 곧 **noise 반복 관측
수집**이며, 서로게이트 층1과 검증 계약의 재료다.

### 출력 2: `database/arrays/{array_id}.npz`

5D bool 배열을 **batch째로** `bool` dtype + `savez_compressed`로 저장한다
(키는 `ARRAY_KEYS` 그대로). x 단위로 쪼개면 파일 수가 폭발하므로 쪼개지 않는다.
catalog의 `(array_id, batch_pos)`가 배열 안의 좌표 역할을 한다.

### 멱등성

- catalog에 이미 있는 `array_id`는 스킵한다.
- 필수 키가 다 모이지 않은 iteration(기록 도중이라 줄이 덜 쌓인 경우)도 스킵하고
  다음 실행에 맡긴다.
- npz를 먼저 쓰고 catalog를 나중에 쓴다. catalog가 완료 표시이므로, 중간에 죽어도
  다시 실행하면 그 iteration부터 다시 처리된다.

### 실행

```bash
python util/ingest.py                      # raw/ -> database/
python util/ingest.py --raw-dir raw --db-dir database
```

단계별 소요 시간(파싱 / 정규화 / npz 쓰기 / catalog 쓰기)을 마지막에 출력한다.
JSON 파싱은 `orjson`을 쓴다(성능상 필수로 검증됨).

## 5. B. MOCKCalculator

`fit()`이 `database/`를 통째로 메모리에 올리는 **스냅샷** 방식. 새 데이터가
들어오면 ingest를 다시 돌리고 `fit()`을 다시 부른다.

### 2층 구조

- **층1 — 관측된 x**: 반복 관측을 **원소별 다수결**로 합쳐 합의(consensus) 배열을
  반환한다. `p_arrays`는 관측 중 True 비율.
- **층2 — 미관측 x**: ordinal 벡터 공간의 **정규화 L1 거리**
  `sum(|a-b| / range) / n_vars`로 k최근접 관측을 찾아 **거리 가중 soft 평균** →
  `p_arrays`(원소별 [0,1]). `p >= 0.5`가 bool 예측.
  최근접 거리가 `trust_dist`를 넘으면 `needs_test=True`
  ("서로게이트를 믿지 말고 TRUE로 실제 평가하라"는 신호).

옵티마이저는 `p_arrays`의 합(`result.score`)을 연속 평가값으로 쓴다. bool로
반올림된 합보다 gradient-free 탐색에서 신호가 매끄럽다.

### 검증 계약

예측이 "관측과 모순되지 않는가"를 판정하는 규칙. 관측이 있는 x에 대해서만 성립한다.

- **배열**
  - 관측 충분(`n_obs >= min_obs`): `관측 교집합 ⊆ 예측 ⊆ 관측 합집합`
  - 부족: 합의 배열의 `erode(d) ⊆ 예측 ⊆ dilate(d)` (거리 `d = dilate_d`, Manhattan)
- **스칼라**
  - 충분: `mean ± z·σ`
  - 부족: `mean ± abs_tol`

파라미터는 전부 `MockConfig` dataclass에 모여 있다.

### 자가 점검

- `self_check()` — 각 관측 x의 합의 배열이 자기 관측 범위를 통과하는지.
  구조적으로 항상 통과해야 하며, 실패하면 ingest나 합의 로직이 깨진 것이다.
- `loo_check()` — leave-one-out. 어떤 x를 이웃 풀에서 빼고 층2로 예측한 뒤 그 x의
  실제 관측으로 검증한다. 층2의 예측력과 `trust_dist` 설정의 타당성을 본다.

### 사용

```python
from MOCKCalculator import MOCKCalculator, MockConfig

mock = MOCKCalculator(db_dir="database", config=MockConfig(k=5, trust_dist=0.05))
mock.fit()

r = mock.predict(candidate_code)
r.score        # p_arrays의 합 = 옵티마이저 평가값
r.needs_test   # True면 TRUE_CALCULATOR로 실제 평가

print(mock.self_check())
print(mock.loo_check())
```

CLI로도 점검할 수 있다.

```bash
python MOCKCalculator.py --db-dir database   # fit + self_check + loo_check 요약
```

## 6. 운영 사이클

```
1. 옵티마이저가 후보 X들을 MOCKCalculator로 평가
2. needs_test=True인 후보만 TRUE_CALCULATOR로 실제 평가
3. TRUE 실행 로그가 raw/에 쌓임
4. python util/ingest.py  (멱등 — 새 run만 처리)
5. mock.fit() 재호출 → 서로게이트가 넓어진 관측 영역을 반영
6. 1로
```

반복할수록 관측 영역이 넓어져 `needs_test` 비율이 떨어지고, 비싼 TRUE 호출이
줄어든다.

## 7. 의존성

```bash
pip install numpy orjson
```

`optimizer.py`(`code_to_ord`, `ord_to_code`)는 이 저장소에 없다. `MOCKCalculator.py`는
이를 import하므로, `optimizer.py`가 import 경로에 없으면 실행되지 않는다.
`util/ingest.py`는 optimizer에 의존하지 않으므로 단독 실행 가능하다.
