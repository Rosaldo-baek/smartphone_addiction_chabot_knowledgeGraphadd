# 패치 적용 완료 안내

기존 두 파일에 KG 연동 코드를 **직접 적용**한 완성본입니다.
아래 파일을 기존 저장소에 그대로 덮어쓰면 됩니다.

## 덮어쓸 파일
- `app_3_5_2_for_rag_v2_2025add.py`              (기존 덮어쓰기, +18줄)
- `script/smart_langgraph_for_3_5_2_v3_2025add.py` (기존 덮어쓰기, +87줄)

## 새로 추가할 파일/폴더
- `script/kg/ontology_builder.py`
- `script/kg/fact_layer.py`
- `script/kg/kg_reasoner.py`
- `script/kg/kg_integration.py`

## requirements.txt
- `networkx==3.4.2` 한 줄 추가됨 (병합본 포함)

## 적용한 패치 (총 7곳)
### app_3_5_2_for_rag_v2_2025add.py
1. import에 `build_knowledge_graph` 추가
2. 세션에 `knowledge_graph` 1회 빌드/캐싱
3. `create_node_functions` 호출에 `knowledge_graph=` 전달

### script/smart_langgraph_for_3_5_2_v3_2025add.py
4. 상단에 KG 모듈 import (실패 시 폴백)
5. `create_node_functions` 시그니처에 `knowledge_graph` 인자 추가
6. `infer_dict_hint` 호출 2곳 → `kg_infer_dict_hint`로 교체 (라우팅)
7. `extract_key_figures` 끝 → KG 수치 적재 추가
8. `validate_answer` → KG 수치 교차검증 추가 (환각 차단)

## 안전장치
모든 패치는 `_KG_AVAILABLE` / `_kg_graph is not None` 가드로 감싸,
KG 모듈 로드 실패 또는 미주입 시 **기존 동작 그대로 폴백**됩니다.

## 상세 변경 내역
- `PATCH_app.diff`       : app 변경 unified diff
- `PATCH_langgraph.diff` : langgraph 변경 unified diff
- `INTEGRATION_GUIDE.md` : 아키텍처/설계 상세 설명

## 검증
```
cd script/kg
python -c "import json,sys; sys.path.insert(0,'.'); \
from kg_integration import build_knowledge_graph; \
g=build_knowledge_graph(json.load(open('../../rag_retrieval_dictionary.json',encoding='utf-8'))); \
print('KG OK:', g['graph'].number_of_nodes(), 'nodes')"
```
