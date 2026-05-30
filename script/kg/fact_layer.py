# =========================================================
# fact_layer.py
# ---------------------------------------------------------
# 역할:
#   5개년 보고서에서 추출한 수치를 "사실 계층" 노드로 적재한다.
#   온톨로지(스키마) 그래프와 동일한 NetworkX 그래프 위에 얹어
#   하나의 통합 KG를 구성한다.
#
# 설계 의도:
#   기존 파이프라인은 매 질의마다 extract_key_figures 노드에서
#   LLM으로 (연도, 대상) → 값 표를 "재추출"했다.
#   → 같은 수치를 매번 다시 뽑으니 느리고, 연도 간 추이 질의가
#     벡터 검색 round-robin 운에 의존했다.
#
#   사실 계층에 한 번 적재해두면:
#     1) 추이/비교 질의를 그래프 조회로 정확히 응답
#     2) LLM이 생성한 답변의 수치를 KG와 교차검증(환각 차단)
#     3) 결측(특정 연도 N/A)을 구조적으로 식별 → 타겟 재검색
#
# 사실 노드 타입:
#   - Observation : 하나의 관측치
#       속성: year, target_group, metric, segment(세분류), value, unit,
#             source_file, page, raw
#
# 사실 엣지 타입:
#   - OBSERVED_FOR   : Observation → TargetGroup (어느 대상의 값인지)
#   - MEASURES       : Observation → Metric (어느 지표인지)
#   - IN_YEAR        : Observation → Year (어느 연도인지)
#   - NEXT_YEAR      : Year(t) → Year(t+1) (시계열 추이 탐색용)
# =========================================================

from __future__ import annotations
import logging
import re
from typing import Optional, List, Dict, Any

import networkx as nx

logger = logging.getLogger(__name__)


def _nid(ntype: str, label: str) -> str:
    """노드 고유 ID 생성 (ontology_builder 와 동일 규칙 유지)."""
    return f"{ntype}:{str(label).strip()}"


# ---------------------------------------------------------
# 값 정규화: "23.5%", "23.5 %", "23.5" → (23.5, '%')
# ---------------------------------------------------------
def _normalize_value(raw: Any) -> tuple[Optional[float], str]:
    """
    수치 문자열에서 float 값과 단위를 분리한다.
    추출 실패(N/A 등) 시 (None, "")을 반환한다.
    """
    if raw is None:
        return None, ""
    s = str(raw).strip()
    if not s or s.upper() in {"N/A", "NA"} or s in {"-", "없음", "미제시", "해당없음"}:
        return None, ""

    # 단위 추정 (% / 점 / 시간 / 분 등)
    unit = ""
    if "%" in s or "퍼센트" in s:
        unit = "%"
    elif "점" in s:
        unit = "점"
    elif "시간" in s:
        unit = "시간"
    elif "분" in s:
        unit = "분"

    # 첫 번째 숫자(소수 포함) 추출
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None, unit
    try:
        return float(m.group()), unit
    except ValueError:
        return None, unit


# ---------------------------------------------------------
# 연도 노드 + 시계열 체인 보장
# ---------------------------------------------------------
def _ensure_year_chain(G: nx.MultiDiGraph, years: List[int]) -> None:
    """
    Year 노드를 만들고 연속 연도 사이에 NEXT_YEAR 엣지를 연결한다.
    추이(trend) 질의 시 연도 순회를 그래프 탐색으로 처리하기 위함.
    """
    sorted_years = sorted(set(int(y) for y in years))
    for y in sorted_years:
        yid = _nid("Year", str(y))
        if not G.has_node(yid):
            G.add_node(yid, ntype="Year", label=str(y), year=y)
    # 인접 연도 연결
    for a, b in zip(sorted_years, sorted_years[1:]):
        if b == a + 1:
            G.add_edge(_nid("Year", str(a)), _nid("Year", str(b)),
                       key="NEXT_YEAR", etype="NEXT_YEAR")


# ---------------------------------------------------------
# 메인: 추출된 수치 표를 사실 계층으로 적재
# ---------------------------------------------------------
def ingest_year_figures(
    G: nx.MultiDiGraph,
    extracted_rows: List[Dict[str, Any]],
    metric: str = "과의존률",
    source_file: str = "",
) -> int:
    """
    extract_key_figures 가 만든 연도별 수치 표를 KG 사실 계층에 적재한다.

    Args:
        G: 통합 지식그래프 (온톨로지 계층이 이미 올라가 있음)
        extracted_rows: [{"연도":2024,"전체":23.5,"유아동":..,"청소년":..,
                          "성인":..,"60대":..}, ...]
        metric: 이 표가 나타내는 지표명 (기본값 과의존률)
        source_file: 출처 파일명 (검증/인용용)

    Returns:
        int: 적재된 Observation 노드 수
    """
    if not extracted_rows:
        return 0

    # 지표 노드 보장
    metric_id = _nid("Metric", metric)
    if not G.has_node(metric_id):
        G.add_node(metric_id, ntype="Metric", label=metric)

    # 표에 등장한 대상 컬럼들 (전체/유아동/청소년/성인/60대)
    target_columns = ["전체", "유아동", "청소년", "성인", "60대"]

    years_seen = []
    count = 0

    for row in extracted_rows:
        if not isinstance(row, dict):
            continue
        try:
            year = int(str(row.get("연도", "")).strip())
        except (ValueError, TypeError):
            continue
        years_seen.append(year)

        for col in target_columns:
            if col not in row:
                continue
            value, unit = _normalize_value(row.get(col))
            if value is None:
                # 결측치도 노드로 남겨두면 "어느 연도가 비었는지" 추적 가능
                # 다만 그래프 비대화를 막기 위해 값 없는 건 스킵하고
                # 결측 사실은 별도 플래그로 관리하는 편이 깔끔하므로 스킵.
                continue

            # Observation 고유 ID: 지표|대상|연도 (동일 키 재적재 시 덮어쓰기)
            obs_label = f"{metric}|{col}|{year}"
            obs_id = _nid("Observation", obs_label)

            G.add_node(
                obs_id, ntype="Observation", label=obs_label,
                year=year, target_group=col, metric=metric,
                segment="전체",  # 세분류(고위험군 등)는 확장 시 사용
                value=value, unit=unit,
                source_file=source_file, raw=str(row.get(col)),
            )

            # 관계 엣지
            tg_id = _nid("TargetGroup", col)
            if G.has_node(tg_id):
                G.add_edge(obs_id, tg_id, key="OBSERVED_FOR", etype="OBSERVED_FOR")
            G.add_edge(obs_id, metric_id, key="MEASURES", etype="MEASURES")

            yid = _nid("Year", str(year))
            if not G.has_node(yid):
                G.add_node(yid, ntype="Year", label=str(year), year=year)
            G.add_edge(obs_id, yid, key="IN_YEAR", etype="IN_YEAR")

            count += 1

    _ensure_year_chain(G, years_seen)
    logger.info("사실 계층 적재: Observation %d개 (지표=%s)", count, metric)
    return count


# ---------------------------------------------------------
# 사실 질의 1) 추이 조회
# ---------------------------------------------------------
def query_trend(
    G: nx.MultiDiGraph,
    target_group: str,
    metric: str = "과의존률",
    years: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    특정 대상의 지표 추이를 연도순으로 반환한다.

    Returns:
        [{"year":2020,"value":..,"unit":"%","source_file":..}, ...] (연도 오름차순)
    """
    results = []
    for nid, data in G.nodes(data=True):
        if data.get("ntype") != "Observation":
            continue
        if data.get("metric") != metric:
            continue
        if data.get("target_group") != target_group:
            continue
        if years and data.get("year") not in years:
            continue
        results.append({
            "year": data.get("year"),
            "value": data.get("value"),
            "unit": data.get("unit", ""),
            "target_group": target_group,
            "source_file": data.get("source_file", ""),
        })
    results.sort(key=lambda r: r["year"])
    return results


# ---------------------------------------------------------
# 사실 질의 2) 연도 단면 비교 (여러 대상 한 연도)
# ---------------------------------------------------------
def query_year_snapshot(
    G: nx.MultiDiGraph,
    year: int,
    metric: str = "과의존률",
) -> Dict[str, Any]:
    """
    한 연도의 모든 대상 값을 dict로 반환한다.
    예: {"전체":23.1, "청소년":40.1, ...}
    """
    snapshot = {}
    for nid, data in G.nodes(data=True):
        if data.get("ntype") != "Observation":
            continue
        if data.get("metric") != metric:
            continue
        if data.get("year") != year:
            continue
        snapshot[data.get("target_group")] = data.get("value")
    return snapshot


# ---------------------------------------------------------
# 사실 질의 3) 결측 연도 탐지
# ---------------------------------------------------------
def find_missing_years(
    G: nx.MultiDiGraph,
    target_group: str,
    requested_years: List[int],
    metric: str = "과의존률",
) -> List[int]:
    """
    요청 연도 중 KG에 값이 없는 연도를 반환한다.
    → 기존 extract_key_figures 의 누락연도 재검색 로직을
       그래프 조회로 정확히 대체.
    """
    have = set()
    for nid, data in G.nodes(data=True):
        if data.get("ntype") == "Observation" \
           and data.get("metric") == metric \
           and data.get("target_group") == target_group:
            have.add(data.get("year"))
    return sorted([y for y in requested_years if y not in have])


# ---------------------------------------------------------
# 검증용: 답변 속 수치가 KG와 일치하는지 교차검증
# ---------------------------------------------------------
def verify_value(
    G: nx.MultiDiGraph,
    target_group: str,
    year: int,
    claimed_value: float,
    metric: str = "과의존률",
    tolerance: float = 0.15,
) -> Dict[str, Any]:
    """
    LLM 답변이 주장한 수치를 KG의 사실값과 비교한다.

    Args:
        claimed_value: 답변에서 추출된 수치
        tolerance: 허용 오차 (절대값). 반올림/표기차 흡수용.

    Returns:
        {"status": "MATCH"|"MISMATCH"|"NOT_IN_KG",
         "kg_value": float|None, "diff": float|None}
    """
    obs_id = _nid("Observation", f"{metric}|{target_group}|{year}")
    if not G.has_node(obs_id):
        return {"status": "NOT_IN_KG", "kg_value": None, "diff": None}

    kg_value = G.nodes[obs_id].get("value")
    if kg_value is None:
        return {"status": "NOT_IN_KG", "kg_value": None, "diff": None}

    diff = abs(kg_value - claimed_value)
    status = "MATCH" if diff <= tolerance else "MISMATCH"
    return {"status": status, "kg_value": kg_value, "diff": round(diff, 3)}
