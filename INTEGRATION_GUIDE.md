# 지식 그래프(KG) 적용 통합 가이드

## 1. 개요

기존 `smartphone_overdependence_chabot_v2`는 ChromaDB 벡터 검색 위에 `rag_retrieval_dictionary.json`(평면 딕셔너리)을 얹어, 동의어 확장·토픽 라우팅·혼동쌍 구분·할루시네이션 방지를 **문자열 부분일치**로 처리하는 구조였습니다.

본 KG 버전은 이 딕셔너리를 **NetworkX 기반 지식 그래프**로 승격하고, 5개년 수치를 **사실 계층**으로 적재하여 다음을 가능하게 합니다.

1. **다중 홉 추론 라우팅** — `쇼츠 → 숏폼(동의어) → T06(토픽)`을 그래프 탐색으로 자동 확장
2. **구조적 추이/비교 질의** — 벡터검색 round-robin 운에 의존하지 않고 그래프에서 정확한 연도별 값 조회
3. **수치 교차검증** — LLM 답변의 수치를 KG 사실값과 대조하여 환각 차단
4. **교차분석 금지 구조화** — `청소년 × 고위험군 세분화` 같은 금지 조합을 `PROHIBITS_CROSS` 엣지로 판정

백엔드는 인메모리(NetworkX)이므로 외부 DB 없이 Streamlit Cloud에 그대로 배포됩니다.

## 2. 파일 구성

```
script/
  smart_langgraph_for_3_5_2_v3_2025add.py   (기존, 일부 패치)
  kg/
    ontology_builder.py    # JSON → 온톨로지 그래프(스키마)
    fact_layer.py          # 연도별 수치 → 사실 계층 + 질의/검증
    kg_reasoner.py         # infer_dict_hint 대체 추론기
    kg_integration.py      # 파이프라인 통합 어댑터(진입점)
```

## 3. 아키텍처

```
[온톨로지 계층 — 스키마]
  CoreConcept(과의존) --HAS_FACTOR--> Factor(조절실패/현저성/문제적_결과)
  Term(쇼츠) --SYNONYM_OF--> Term(숏폼) --ROUTES_TO--> Topic(T06)
  Term(이용률) --CONFUSED_WITH--> Term(이용정도)
  Term(청소년) --PROHIBITS_CROSS--> Term(고위험군)

[사실 계층 — 데이터]
  Observation(과의존률|청소년|2024)
    --IN_YEAR--> Year(2024)
    --OBSERVED_FOR--> TargetGroup(청소년)
    --MEASURES--> Metric(과의존률)
  Year(2023) --NEXT_YEAR--> Year(2024)
```

두 계층은 **하나의 NetworkX MultiDiGraph**에 통합됩니다.

## 4. 기존 코드 수정 지점 (3곳 + 임포트)

### 4-1. import 추가 (`smart_langgraph...py` 상단)

```python
import os, sys
_KG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kg")
if _KG_DIR not in sys.path:
    sys.path.insert(0, _KG_DIR)

from kg_integration import (
    build_knowledge_graph,
    kg_infer_dict_hint,
    kg_ingest_from_state,
    kg_validate_answer,
    kg_build_trend_context,
)
```

### 4-2. `create_node_functions` 시그니처에 KG 주입 (기존 852행)

```python
# 변경 전
def create_node_functions(vectorstore, llms, status_callback, rag_dict_index):

# 변경 후 — knowledge_graph 인자 추가
def create_node_functions(vectorstore, llms, status_callback, rag_dict_index,
                          knowledge_graph=None):
    _kg_graph = (knowledge_graph or {}).get("graph")
    _kg_reasoner = (knowledge_graph or {}).get("reasoner")
```

### 4-3. `infer_dict_hint` 호출 → KG 추론으로 교체 (기존 1663행, 1820행)

두 곳 모두 동일 패턴으로 교체합니다. KG 결과가 비면 기존 함수로 자동 폴백되므로 안전합니다.

```python
# 변경 전 (1663행 예시)
dict_hint = infer_dict_hint(user_input, context_text=context_text,
                            rag_dict_index=rag_dict_index)

# 변경 후
dict_hint = kg_infer_dict_hint(
    user_input,
    context_text=context_text,
    reasoner=_kg_reasoner,           # KG 추론기 (None이면 폴백만)
    fallback_fn=infer_dict_hint,     # 기존 함수 그대로 폴백
    rag_dict_index=rag_dict_index,
)
```

### 4-4. `extract_key_figures` 끝에 수치 적재 추가 (기존 2531행 직후)

```python
state["extracted_figures_json"] = {"연도별_수치": ordered_rows}
state["year_extractions"] = ordered_rows

# --- [KG 추가] 추출 수치를 사실 계층에 적재 ---
if _kg_graph is not None:
    try:
        kg_ingest_from_state(_kg_graph, state, metric="과의존률")
    except Exception as e:
        logger.warning("KG 수치 적재 실패: %s", e)
```

### 4-5. `validate_answer` 에 수치 교차검증 추가 (기존 2638행 함수 내부)

검증 결과가 PASS여도 KG와 수치가 어긋나면 재생성(FAIL)으로 강등합니다.

```python
def validate_answer(state: GraphState) -> GraphState:
    # ... 기존 검증 로직으로 validation_result 결정 ...

    # --- [KG 추가] 답변 수치를 KG 사실값과 교차검증 ---
    if _kg_graph is not None:
        try:
            draft = state.get("draft_answer") or state.get("final_answer") or ""
            kg_check = kg_validate_answer(_kg_graph, draft, metric="과의존률")
            state.setdefault("debug_info", {})["kg_value_check"] = kg_check
            if kg_check.get("has_mismatch"):
                # KG와 명백히 어긋난 수치 → 근거 부족으로 재검색 유도
                state["validation_result"] = "FAIL_NO_EVIDENCE"
                state["validation_reason"] = (
                    "KG 사실값과 답변 수치 불일치: " + str(kg_check["mismatches"])
                )
        except Exception as e:
            logger.warning("KG 검증 실패: %s", e)

    return state
```

### 4-6. (선택) 추이 컨텍스트 주입 — `extract_key_figures` 또는 `generate_answer` 앞

단일 대상 다년도 질의일 때 KG에서 정확한 추이표를 만들어 컨텍스트 상단에 주입하면, 벡터검색 누락에도 정확한 추이를 보장합니다.

```python
target = (state.get("dict_hint") or {}).get("target_group")
years = (state.get("plan") or {}).get("years", [])
if _kg_graph is not None and target and len(years) >= 3:
    trend_md = kg_build_trend_context(_kg_graph, target, years, "과의존률")
    if trend_md:
        state["context"] = trend_md + "\n\n---\n\n" + (state.get("context") or "")
```

## 5. 앱 진입점 수정 (`app_3_5_2_for_rag_v2_2025add.py`)

`create_node_functions` 호출 시 KG를 만들어 전달합니다.

```python
from smart_langgraph_for_3_5_2_v3_2025add import (
    ..., build_knowledge_graph,   # 추가 export
)

# 세션 상태에 KG 캐싱 (1회 빌드)
if "knowledge_graph" not in st.session_state:
    st.session_state.knowledge_graph = build_knowledge_graph(st.session_state.rag_dict)

# 노드 함수 생성 시 KG 전달
st.session_state.node_functions = create_node_functions(
    vectorstore, llms, status_callback,
    st.session_state.rag_dict_index,
    knowledge_graph=st.session_state.knowledge_graph,   # 추가
)
```

> `build_knowledge_graph`를 `smart_langgraph...py`에서 re-export 하거나, 앱에서 `kg_integration`을 직접 import 해도 됩니다.

## 6. requirements

기존 `requirements.txt`에 한 줄 추가:

```
networkx==3.4.2
```

## 7. 동작 검증 결과 (test_kg.py)

| 테스트 | 내용 | 결과 |
|--------|------|------|
| A | 동의어/토픽 라우팅 (쇼츠→숏폼→T06) | topic=T06, anchor=[쇼츠,릴스,틱톡,숏폼…] |
| B | 혼동쌍 (이용률↔이용정도) | avoid=[이용정도], 구분 힌트 생성 |
| C/G | 교차분석 금지 (청소년×고위험군) | PROHIBITS_CROSS 위반 감지 |
| D | 추이 질의 (그래프 조회) | 2020~2024 정확한 연도별 값 |
| E | 결측 연도 탐지 | 유아동 2024 결측 식별 |
| F | 수치 교차검증 | MATCH/MISMATCH/NOT_IN_KG 정확 판정 |

## 8. 점진적 도입 전략

KG는 **폴백 안전장치**가 내장되어 있습니다.

- `knowledge_graph=None`이면 모든 KG 함수가 기존 동작으로 폴백 → 한 번에 다 바꾸지 않고 4-3(라우팅)만 먼저 적용해 효과를 관찰한 뒤, 4-4/4-5(적재·검증)를 순차 도입할 수 있습니다.
- 사실 계층은 질의가 쌓일수록 채워지므로, 초기엔 온톨로지 라우팅 이점만 보다가 운영 중 추이/검증 효과가 누적됩니다.
- (운영 팁) 사실 계층을 매 세션 새로 채우는 대신, 배치로 5개년 표를 한 번 적재해 `nx.write_gpickle`로 저장해두면 콜드스타트가 사라집니다.

## 9. 한계와 후속 과제

- 수치 적재는 `extract_key_figures`(과의존률 표) 출력에 의존합니다. 다른 지표(이용시간·요인점수 등)는 `metric` 인자를 바꿔 동일 방식으로 적재하도록 확장 필요.
- `kg_validate_answer`의 수치 추출은 휴리스틱(정규식)이라 표/문장이 모호하면 검증을 건너뜁니다. 오탐 방지를 위해 MISMATCH만 강하게 다루고 NOT_IN_KG는 무시합니다.
- 교차분석 금지쌍 중 "성별 × 연령대별" 같은 **변수 축 조합**은 질의의 구체 토큰과 직접 매칭되지 않아, 대상↔축 의미 매핑을 `_augment_domain_knowledge`에서 명시적으로 보강했습니다. 신규 금지 패턴은 이 함수에 추가하면 됩니다.
