# =========================================================
# kg_integration.py
# ---------------------------------------------------------
# 역할:
#   앞서 만든 3개 모듈(ontology_builder / fact_layer / kg_reasoner)을
#   기존 LangGraph 파이프라인(smart_langgraph_for_3_5_2_v3_2025add.py)에
#   "최소 침습"으로 끼워넣기 위한 어댑터/진입점.
#
# 통합 지점 (3곳):
#   1) build_knowledge_graph()  : 앱 시작 시 1회 호출해 통합 KG 생성/캐싱
#   2) kg_infer_dict_hint()     : 기존 infer_dict_hint() 대체 (drop-in)
#   3) kg_ingest_from_state()   : extract_key_figures 직후 수치를 KG에 적재
#      kg_validate_answer()     : validate_answer 단계에서 수치 교차검증
#
# 설계 원칙:
#   - 기존 함수 시그니처/반환 스키마를 깨지 않는다.
#   - KG가 빈 결과를 내면 기존 규칙기반 결과로 폴백한다(안전).
# =========================================================

from __future__ import annotations
import json
import logging
import re
from typing import Optional, List, Dict, Any

import networkx as nx

from ontology_builder import build_ontology_graph
from fact_layer import (
    ingest_year_figures, query_trend, query_year_snapshot,
    find_missing_years, verify_value,
)
from kg_reasoner import KGReasoner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 1) 통합 KG 빌더 (앱 시작 시 1회)
# ---------------------------------------------------------
def build_knowledge_graph(rag_dict: dict) -> Dict[str, Any]:
    """
    온톨로지 계층 그래프를 구축하고 추론기를 함께 반환한다.
    사실 계층은 질의가 진행되며 점진적으로 채워진다.

    Returns:
        {"graph": nx.MultiDiGraph, "reasoner": KGReasoner}
    """
    G = build_ontology_graph(rag_dict)
    reasoner = KGReasoner(G)
    return {"graph": G, "reasoner": reasoner}


# ---------------------------------------------------------
# 2) infer_dict_hint 대체 (drop-in)
#    기존 plan_search 는 infer_dict_hint(text, context, rag_dict_index)를
#    호출한다. 같은 시그니처로 KG 추론 결과를 돌려주되, 기존 규칙기반
#    결과와 병합해 누락을 보완한다.
# ---------------------------------------------------------
def kg_infer_dict_hint(
    text: str,
    context_text: str = "",
    reasoner: Optional[KGReasoner] = None,
    fallback_fn=None,
    rag_dict_index: dict = None,
) -> dict:
    """
    KG 기반 dict_hint 생성. KG 결과가 비면 기존 함수로 폴백/병합한다.

    Args:
        text: 사용자 질문
        context_text: 이전 대화 컨텍스트
        reasoner: KGReasoner 인스턴스 (없으면 폴백만 수행)
        fallback_fn: 기존 infer_dict_hint 함수 (병합용)
        rag_dict_index: 폴백 함수에 전달할 인덱스
    """
    # 폴백(기존 규칙기반) 결과 먼저 확보
    base = {}
    if fallback_fn is not None:
        try:
            base = fallback_fn(text, context_text, rag_dict_index) or {}
        except Exception as e:
            logger.warning("기존 infer_dict_hint 폴백 실패: %s", e)
            base = {}

    if reasoner is None:
        return base

    # KG 추론 결과
    try:
        kg = reasoner.infer_hint(text, context_text)
    except Exception as e:
        logger.warning("KG 추론 실패, 폴백 사용: %s", e)
        return base

    # 병합: KG 우선, 빈 값은 폴백으로 보완
    def _pick(key, default):
        kv = kg.get(key)
        if kv:  # truthy(비어있지 않음)면 KG 채택
            return kv
        return base.get(key, default)

    merged = {
        "is_rag_like": kg.get("is_rag_like") or base.get("is_rag_like", False),
        "topic_code": _pick("topic_code", ""),
        "target_group": _pick("target_group", ""),
        # 앵커/회피어는 합집합으로 (더 풍부한 검색 단서 확보)
        "anchor_terms": _merge_lists(kg.get("anchor_terms"), base.get("anchor_terms")),
        "avoid_terms": _merge_lists(kg.get("avoid_terms"), base.get("avoid_terms")),
        "needs_appendix_table": kg.get("needs_appendix_table")
                                 or base.get("needs_appendix_table", False),
        "scope_warnings": _merge_lists(kg.get("scope_warnings"), base.get("scope_warnings")),
        # KG 전용 필드 (다운스트림 검증에서 활용)
        "prohibited_cross": kg.get("prohibited_cross", []),
        "confused_pairs": kg.get("confused_pairs", []),
        "kg_matched_nodes": kg.get("matched_nodes", []),
    }
    return merged


def _merge_lists(a, b):
    """두 리스트를 순서 보존하며 중복 없이 합친다."""
    out, seen = [], set()
    for lst in (a or []), (b or []):
        for x in lst:
            key = json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
            if key not in seen:
                seen.add(key)
                out.append(x)
    return out


# ---------------------------------------------------------
# 3) 수치 적재: extract_key_figures 결과 → KG 사실 계층
# ---------------------------------------------------------
def kg_ingest_from_state(graph: nx.MultiDiGraph, state: dict,
                         metric: str = "과의존률") -> int:
    """
    LangGraph state 의 extracted_figures_json 을 KG에 적재한다.
    extract_key_figures 노드 끝에서 호출하면 된다.

    Returns:
        적재된 Observation 수
    """
    figures = state.get("extracted_figures_json") or {}
    rows = figures.get("연도별_수치", [])
    if not rows:
        return 0

    # 출처 파일명 (plan에서 확보)
    plan = state.get("plan") or {}
    files = plan.get("file_name_filters", [])
    source = files[0] if len(files) == 1 else "복수연도"

    return ingest_year_figures(graph, rows, metric=metric, source_file=source)


# ---------------------------------------------------------
# 4) 답변 수치 교차검증: validate_answer 단계 보강
# ---------------------------------------------------------
# 답변 텍스트에서 "(연도) ... (대상) ... NN.N%" 패턴을 추출하기 위한 정규식
_YEAR_RE = re.compile(r"(20[2][0-5])\s*년")
_TARGET_TOKENS = ["전체", "유아동", "청소년", "성인", "60대"]
_VALUE_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")


def kg_validate_answer(graph: nx.MultiDiGraph, answer: str,
                       metric: str = "과의존률",
                       tolerance: float = 0.15) -> Dict[str, Any]:
    """
    답변 본문에서 (연도, 대상, 수치)를 추출해 KG 사실값과 대조한다.
    명백한 MISMATCH가 있으면 검증 실패 신호를 돌려준다.

    주의: 휴리스틱 추출이므로 NOT_IN_KG는 '검증 불가'로만 취급하고
          MISMATCH만 강한 경고로 다룬다(오탐 방지).

    Returns:
        {
          "checked": int,         # 검증 시도한 수치 개수
          "mismatches": [..],     # KG와 어긋난 항목
          "has_mismatch": bool,
        }
    """
    results = {"checked": 0, "mismatches": [], "has_mismatch": False}
    if not answer:
        return results

    # 표 형태(| 2024 | ... | 42.6 |)와 문장 형태를 모두 대략 커버하기 위해
    # 줄 단위로 연도/대상/값을 동시에 포함하는 라인을 검사한다.
    for line in answer.split("\n"):
        years = _YEAR_RE.findall(line)
        values = _VALUE_RE.findall(line)
        if not years or not values:
            continue
        targets_in_line = [t for t in _TARGET_TOKENS if t in line]
        if not targets_in_line:
            continue

        # 한 줄에 연도/대상/값이 1:1:1로 명확할 때만 검증 (모호하면 스킵)
        if len(years) == 1 and len(targets_in_line) == 1 and len(values) == 1:
            year = int(years[0])
            tg = targets_in_line[0]
            val = float(values[0])
            check = verify_value(graph, tg, year, val, metric, tolerance)
            results["checked"] += 1
            if check["status"] == "MISMATCH":
                results["mismatches"].append({
                    "year": year, "target_group": tg,
                    "claimed": val, "kg_value": check["kg_value"],
                    "diff": check["diff"],
                })

    results["has_mismatch"] = len(results["mismatches"]) > 0
    return results


# ---------------------------------------------------------
# 5) 추이 질의 헬퍼: 답변 생성 전 KG에서 정확한 추이표 확보
#    plan.years 가 3개 이상이고 단일 대상이면, 벡터검색에 의존하기 전에
#    KG에서 정확한 추이를 끌어와 컨텍스트 상단에 주입할 수 있다.
# ---------------------------------------------------------
def kg_build_trend_context(graph: nx.MultiDiGraph, target_group: str,
                           years: List[int], metric: str = "과의존률") -> str:
    """
    KG에 적재된 사실로 추이 마크다운 표를 만든다.
    아직 적재 전이면 빈 문자열(폴백→기존 벡터검색)을 반환.
    """
    if not target_group or not years:
        return ""
    trend = query_trend(graph, target_group, metric, years)
    if not trend:
        return ""

    lines = [f"[KG 사실 기반 {target_group} {metric} 추이 — 우선 참조]",
             "| 연도 | 값 |", "|------|-----|"]
    for t in trend:
        unit = t.get("unit", "")
        lines.append(f"| {t['year']} | {t['value']}{unit} |")
    return "\n".join(lines)
