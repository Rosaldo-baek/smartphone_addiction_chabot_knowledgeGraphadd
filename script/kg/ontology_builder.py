# =========================================================
# ontology_builder.py
# ---------------------------------------------------------
# 역할:
#   기존 rag_retrieval_dictionary.json(평면 딕셔너리)을
#   NetworkX 기반 "온톨로지 계층(스키마) 지식 그래프"로 승격한다.
#
# 설계 의도:
#   - 기존 코드는 `if "이용률" in q` 같은 하드코딩 문자열 매칭과
#     `pat.lower() in q_low` 부분일치로 라우팅/동의어/혼동쌍을 처리했다.
#   - 이를 그래프 노드/엣지로 바꾸면 "다중 홉 추론"이 가능해진다.
#     예) 사용자가 "쇼츠 조절이 안돼요" → (쇼츠=숏폼 SYNONYM_OF) →
#         (숏폼 BELONGS_TO T06) → (T06 RELATED_TO 조절실패=과의존 3요인) 등
#         관계를 따라가며 토픽/앵커를 자동 확장한다.
#
# 노드 타입(node 'ntype' 속성):
#   - Term          : 일반 용어/동의어 토큰 (검색 앵커 후보)
#   - CoreConcept   : 핵심 개념(과의존, 과의존위험군 등)
#   - Factor        : 과의존 3요인(조절실패/현저성/문제적_결과)
#   - TargetGroup   : 조사 대상(유아동/청소년/성인/60대 등)
#   - Topic         : 토픽 분류(T01~T11)
#   - Subtopic      : 토픽 하위 항목
#   - Metric        : 지표(이용률/이용정도/과의존률 등)
#   - Banner        : 통계표 배너(B1~B6: 전체/과의존수준별/연령대별 등)
#   - Rule          : 할루시네이션 방지/혼동방지 규칙
#
# 엣지 타입(edge 'etype' 속성):
#   - SYNONYM_OF     : 동의어 관계 (양방향 추가)
#   - NOT_SYNONYM_OF : 비동의어(혼동 주의) 관계
#   - HAS_SUBTOPIC   : Topic → Subtopic
#   - HAS_FACTOR     : CoreConcept(과의존) → Factor
#   - ALIAS_OF       : 별칭 → 표준 TargetGroup
#   - ROUTES_TO      : Term/패턴 → Topic (라우팅 힌트)
#   - CONFUSED_WITH  : 혼동 가능 쌍 (disambiguation)
#   - PROHIBITS_CROSS: 교차분석 금지 조합 (환각 방지 핵심)
#   - SCOPED_BY      : Metric/Subtopic → Banner (어느 배너에서만 존재하는지)
# =========================================================

from __future__ import annotations
import json
import logging
import re
from typing import Optional

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# 라벨 정제
#   실제 JSON에는 "과의존위험군 비율(T01)" 처럼 괄호 토픽코드가 붙거나
#   "3요인 조절실패(T03)" 같이 접두 수식어가 붙은 라벨이 많다.
#   이를 매칭 가능한 핵심 토큰으로 정제하고, 원문도 별칭으로 함께 보존한다.
# ---------------------------------------------------------
def _clean_label(raw: str) -> str:
    """괄호 안 토픽코드/부연설명을 제거해 핵심 토큰만 남긴다."""
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    # 괄호류 제거: (T01), （...）, [..]
    s = re.sub(r"[\(\（\[][^\)\）\]]*[\)\）\]]", "", s)
    return s.strip()


# ---------------------------------------------------------
# 노드 ID 생성 헬퍼
#   동일 라벨이라도 타입이 다르면 충돌하지 않도록 "타입:라벨" 형태로 ID를 만든다.
# ---------------------------------------------------------
def _nid(ntype: str, label: str) -> str:
    """노드 고유 ID를 생성한다. (예: 'Term:쇼츠')"""
    return f"{ntype}:{str(label).strip()}"


def _add_node(G: nx.MultiDiGraph, ntype: str, label: str, **attrs) -> str:
    """
    노드를 추가(또는 속성 병합)하고 노드 ID를 반환한다.
    이미 존재하면 새 속성만 갱신한다.
    """
    nid = _nid(ntype, label)
    if G.has_node(nid):
        # 기존 노드면 속성만 업데이트 (라벨/타입은 유지)
        for k, v in attrs.items():
            if v is not None:
                G.nodes[nid][k] = v
    else:
        G.add_node(nid, ntype=ntype, label=str(label).strip(), **attrs)
    return nid


def _add_edge(G: nx.MultiDiGraph, src: str, dst: str, etype: str, **attrs) -> None:
    """
    방향 엣지를 추가한다. MultiDiGraph이므로 동일 노드쌍에
    서로 다른 etype의 엣지가 공존할 수 있다.
    key=etype 로 지정해 같은 타입 엣지의 중복 생성을 방지한다.
    """
    G.add_edge(src, dst, key=etype, etype=etype, **attrs)


# =========================================================
# 메인 빌더
# =========================================================
def build_ontology_graph(rag_dict: dict) -> nx.MultiDiGraph:
    """
    RAG Dictionary(JSON)를 받아 온톨로지 지식그래프를 구축한다.

    Args:
        rag_dict: rag_retrieval_dictionary.json 을 로드한 dict

    Returns:
        nx.MultiDiGraph: 온톨로지 계층 그래프
    """
    G = nx.MultiDiGraph()
    rag_dict = rag_dict or {}

    _build_core_definitions(G, rag_dict)
    _build_target_groups(G, rag_dict)
    _build_topic_taxonomy(G, rag_dict)
    _build_routing_patterns(G, rag_dict)
    _build_disambiguation(G, rag_dict)
    _build_banner_structure(G, rag_dict)
    _build_hallucination_rules(G, rag_dict)
    _augment_domain_knowledge(G)  # 기존 infer_dict_hint 하드코딩 지식 보강

    logger.info(
        "온톨로지 그래프 구축 완료: 노드 %d개, 엣지 %d개",
        G.number_of_nodes(), G.number_of_edges()
    )
    return G


# ---------------------------------------------------------
# 1) core_definitions: 핵심 개념/동의어/3요인
# ---------------------------------------------------------
def _build_core_definitions(G: nx.MultiDiGraph, rag_dict: dict) -> None:
    core = rag_dict.get("core_definitions", {}) or {}
    for concept, body in core.items():
        if concept.startswith("_") or not isinstance(body, dict):
            continue

        # 핵심 개념 노드
        concept_id = _add_node(
            G, "CoreConcept", concept,
            definition=body.get("definition", "")
        )

        # 동의어 → SYNONYM_OF (양방향: 검색 시 어느 쪽으로 들어와도 확장되도록)
        for syn in body.get("synonyms", []) or []:
            term_id = _add_node(G, "Term", syn)
            _add_edge(G, term_id, concept_id, "SYNONYM_OF")
            _add_edge(G, concept_id, term_id, "SYNONYM_OF")

        # 비동의어 → NOT_SYNONYM_OF (혼동 주의 신호)
        for nsyn in body.get("NOT_synonyms", []) or []:
            term_id = _add_node(G, "Term", nsyn)
            _add_edge(G, concept_id, term_id, "NOT_SYNONYM_OF")

        # 과의존 3요인 처리 (조절실패/현저성/문제적_결과)
        factors = body.get("과의존_3요인") if concept == "과의존" else None
        # 일부 JSON에서는 3요인이 과의존 하위가 아니라 별도 키일 수 있어 방어적으로 처리
        if isinstance(body.get("과의존_3요인"), dict):
            factors = body.get("과의존_3요인")
        if isinstance(factors, dict):
            for fname, fbody in factors.items():
                if fname.startswith("_") or not isinstance(fbody, dict):
                    continue
                factor_id = _add_node(
                    G, "Factor", fname,
                    definition=fbody.get("definition", "")
                )
                _add_edge(G, concept_id, factor_id, "HAS_FACTOR")
                # 요인별 키워드도 Term으로 연결
                for kw in fbody.get("keywords", []) or []:
                    kw_id = _add_node(G, "Term", kw)
                    _add_edge(G, kw_id, factor_id, "SYNONYM_OF")

    # core_definitions 최상위에 과의존_3요인이 따로 있는 경우도 커버
    top_factors = core.get("과의존_3요인")
    if isinstance(top_factors, dict):
        concept_id = _add_node(G, "CoreConcept", "과의존")
        for fname, fbody in top_factors.items():
            if fname.startswith("_") or not isinstance(fbody, dict):
                continue
            factor_id = _add_node(
                G, "Factor", fname, definition=fbody.get("definition", "")
            )
            _add_edge(G, concept_id, factor_id, "HAS_FACTOR")
            for kw in fbody.get("keywords", []) or []:
                kw_id = _add_node(G, "Term", kw)
                _add_edge(G, kw_id, factor_id, "SYNONYM_OF")


# ---------------------------------------------------------
# 2) target_groups: 조사 대상 + 별칭 + 교차집단
# ---------------------------------------------------------
def _build_target_groups(G: nx.MultiDiGraph, rag_dict: dict) -> None:
    targets = rag_dict.get("target_groups", {}) or {}
    for tg_name, tg_body in targets.items():
        if tg_name.startswith("_"):
            continue

        # cross_cutting_groups(학부모 등)는 별도 처리
        if tg_name == "cross_cutting_groups" and isinstance(tg_body, dict):
            for cc_name, cc_body in tg_body.items():
                if cc_name.startswith("_"):
                    continue
                definition = cc_body.get("definition", "") if isinstance(cc_body, dict) else str(cc_body)
                _add_node(G, "TargetGroup", cc_name, is_cross_cutting=True,
                          definition=definition)
            continue

        if not isinstance(tg_body, dict):
            continue

        tg_id = _add_node(
            G, "TargetGroup", tg_name,
            age_range=tg_body.get("age_range", ""),
            respondent=tg_body.get("respondent", ""),
            scale_type=tg_body.get("scale_type", ""),
        )

        # 별칭(also_called) → ALIAS_OF
        for alias in tg_body.get("also_called", []) or []:
            alias_id = _add_node(G, "Term", alias)
            _add_edge(G, alias_id, tg_id, "ALIAS_OF")


# ---------------------------------------------------------
# 3) topic_taxonomy: 토픽(T01~T11) + 하위토픽 + 대표질의
# ---------------------------------------------------------
def _build_topic_taxonomy(G: nx.MultiDiGraph, rag_dict: dict) -> None:
    topics = rag_dict.get("topic_taxonomy", {}) or {}
    for tkey, tbody in topics.items():
        if tkey.startswith("_") or not isinstance(tbody, dict):
            continue

        # 토픽 코드는 "T01_과의존_현황_추이" → 코드 'T01' 추출
        tcode = tkey.split("_")[0]
        topic_id = _add_node(
            G, "Topic", tkey,
            code=tcode,
            description=tbody.get("description", ""),
        )

        # 하위 토픽 → HAS_SUBTOPIC
        subtopics = tbody.get("subtopics", {}) or {}
        for sub_name, sub_val in subtopics.items():
            if sub_name.startswith("_"):
                continue
            sub_id = _add_node(G, "Subtopic", sub_name)
            _add_edge(G, topic_id, sub_id, "HAS_SUBTOPIC")

        # 대표 질의(typical_queries)의 토큰을 Term으로 연결 → ROUTES_TO
        # (질의 패턴 학습 없이도 대표질의 키워드로 라우팅 단서 확보)
        for tq in tbody.get("typical_queries", []) or []:
            tq_id = _add_node(G, "Term", tq, is_typical_query=True)
            _add_edge(G, tq_id, topic_id, "ROUTES_TO", weight=0.5)


# ---------------------------------------------------------
# 4) query_routing_guide: 명시적 라우팅 패턴 → ROUTES_TO
# ---------------------------------------------------------
def _build_routing_patterns(G: nx.MultiDiGraph, rag_dict: dict) -> None:
    routing = (rag_dict.get("query_routing_guide", {}) or {}).get("patterns", []) or []
    for item in routing:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("query_pattern", "") or "").strip()
        primary = str(item.get("primary_topic", "") or "").strip()
        secondary = str(item.get("secondary_topic", "") or "").strip()
        if not pattern or not primary:
            continue

        # 패턴은 콤마로 여러 키워드가 묶여 있을 수 있음
        for raw in pattern.split(","):
            term = raw.strip()
            if not term:
                continue
            term_id = _add_node(G, "Term", term)
            # primary_topic 은 'T05' 같은 코드일 수 있고 'T05_콘텐츠...' 풀네임일 수도 있음
            primary_topic_id = _resolve_topic_node(G, primary)
            if primary_topic_id:
                _add_edge(G, term_id, primary_topic_id, "ROUTES_TO",
                          weight=1.0, role="primary")
            if secondary:
                secondary_topic_id = _resolve_topic_node(G, secondary)
                if secondary_topic_id:
                    _add_edge(G, term_id, secondary_topic_id, "ROUTES_TO",
                              weight=0.4, role="secondary")


def _resolve_topic_node(G: nx.MultiDiGraph, topic_ref: str) -> Optional[str]:
    """
    'T05' 또는 'T05_콘텐츠_이용현황' 형태의 참조를 실제 Topic 노드 ID로 해석한다.
    """
    topic_ref = topic_ref.strip()
    # 풀네임으로 바로 존재하면 사용
    direct = _nid("Topic", topic_ref)
    if G.has_node(direct):
        return direct
    # 코드(T05)로 들어온 경우 code 속성으로 검색
    code = topic_ref.split("_")[0]
    for nid, data in G.nodes(data=True):
        if data.get("ntype") == "Topic" and data.get("code") == code:
            return nid
    return None


# ---------------------------------------------------------
# 5) disambiguation_rules: 혼동쌍 → CONFUSED_WITH
# ---------------------------------------------------------
def _build_disambiguation(G: nx.MultiDiGraph, rag_dict: dict) -> None:
    rules = rag_dict.get("disambiguation_rules", {}) or {}
    for rkey, rbody in rules.items():
        if rkey.startswith("_") or not isinstance(rbody, dict):
            continue
        pair = rbody.get("confusable_pair", []) or []
        distinction = rbody.get("distinction", "")
        routing_hint = rbody.get("routing_hint", "")
        if len(pair) < 2:
            continue

        # 쌍 내 모든 조합을 CONFUSED_WITH 로 연결 (3개짜리 쌍도 커버)
        # 실제 라벨이 "과의존위험군 비율(T01)" 형태이므로 정제 토큰으로 노드를 만들되,
        # 원문도 별칭(Term)으로 보존해 어느 쪽으로 매칭돼도 잡히게 한다.
        term_ids = []
        for p in pair:
            clean = _clean_label(p)
            if not clean:
                continue
            cid = _add_node(G, "Term", clean)
            term_ids.append(cid)
            # 원문 라벨이 정제본과 다르면 별칭으로 추가 연결
            if p.strip() != clean:
                raw_id = _add_node(G, "Term", p.strip())
                _add_edge(G, raw_id, cid, "SYNONYM_OF")
                _add_edge(G, cid, raw_id, "SYNONYM_OF")
        for i in range(len(term_ids)):
            for j in range(i + 1, len(term_ids)):
                _add_edge(G, term_ids[i], term_ids[j], "CONFUSED_WITH",
                          rule_id=rkey, distinction=distinction,
                          routing_hint=routing_hint)
                _add_edge(G, term_ids[j], term_ids[i], "CONFUSED_WITH",
                          rule_id=rkey, distinction=distinction,
                          routing_hint=routing_hint)


# ---------------------------------------------------------
# 6) stat_table_banner_structure: 배너(B1~B6) + 결측 패턴
# ---------------------------------------------------------
def _build_banner_structure(G: nx.MultiDiGraph, rag_dict: dict) -> None:
    banner_info = rag_dict.get("stat_table_banner_structure", {}) or {}
    hierarchy = banner_info.get("banner_hierarchy", {}) or {}
    for bkey, bbody in hierarchy.items():
        if bkey.startswith("_") or not isinstance(bbody, dict):
            continue
        banner_id = _add_node(
            G, "Banner", bkey,
            description=bbody.get("description", ""),
            categories=bbody.get("categories", []),
            note=bbody.get("note", ""),
        )
        # 배너 카테고리(연령대/성별/학령 등)를 Term으로 연결 → SCOPED_BY
        for cat in bbody.get("categories", []) or []:
            cat_id = _add_node(G, "Term", cat)
            _add_edge(G, cat_id, banner_id, "SCOPED_BY")


# ---------------------------------------------------------
# 7) hallucination_prevention: 교차분석 금지 → PROHIBITS_CROSS
#    (환각 방지의 핵심. 그래프에 금지 엣지로 박아두고 검증 단계에서 조회)
# ---------------------------------------------------------
def _build_hallucination_rules(G: nx.MultiDiGraph, rag_dict: dict) -> None:
    h_rules = rag_dict.get("hallucination_prevention", {}) or {}
    for rkey, rbody in h_rules.items():
        if rkey.startswith("_") or not isinstance(rbody, dict):
            continue

        rule_id = _add_node(
            G, "Rule", rkey,
            title=rbody.get("title", ""),
            detail=rbody.get("detail", ""),
            retrieval_rule=rbody.get("retrieval_rule", ""),
        )

        # H01: 교차분석 금지 조합을 PROHIBITS_CROSS 엣지로 변환
        prohibited = rbody.get("prohibited_combinations", []) or []
        for combo in prohibited:
            # combo 예: "유아동 × 과의존위험군" 같은 문자열
            parts = _split_combo(combo)
            if len(parts) >= 2:
                a_id = _add_node(G, "Term", parts[0])
                b_id = _add_node(G, "Term", parts[1])
                _add_edge(G, a_id, b_id, "PROHIBITS_CROSS",
                          rule_id=rkey, raw=combo)
                _add_edge(G, b_id, a_id, "PROHIBITS_CROSS",
                          rule_id=rkey, raw=combo)


# ---------------------------------------------------------
# 8) 도메인 지식 보강
#    기존 infer_dict_hint() 가 if문으로 하드코딩하던 지식 중
#    JSON에 노드로 표현되지 않은 것들을 그래프에 명시적으로 주입한다.
#    (JSON 갱신과 무관하게 핵심 동의어/혼동쌍을 안정적으로 보장)
# ---------------------------------------------------------
def _augment_domain_knowledge(G: nx.MultiDiGraph) -> None:
    # (1) 숏폼 계열 동의어: 쇼츠/릴스/틱톡 → 숏폼, 숏폼은 T06으로 라우팅
    shortform_id = _add_node(G, "Term", "숏폼")
    t06 = _resolve_topic_node(G, "T06")
    if t06:
        _add_edge(G, shortform_id, t06, "ROUTES_TO", weight=1.0, role="primary")
    for syn in ["쇼츠", "릴스", "틱톡", "쇼트폼", "short-form"]:
        sid = _add_node(G, "Term", syn)
        _add_edge(G, sid, shortform_id, "SYNONYM_OF")
        _add_edge(G, shortform_id, sid, "SYNONYM_OF")
        if t06:
            _add_edge(G, sid, t06, "ROUTES_TO", weight=0.8, role="primary")

    # (2) 핵심 혼동쌍: 이용률 ↔ 이용정도 (원본 infer_dict_hint의 대표 케이스)
    rate_id = _add_node(G, "Term", "이용률")
    degree_id = _add_node(G, "Term", "이용정도")
    _add_edge(G, rate_id, degree_id, "CONFUSED_WITH",
              rule_id="domain_aug",
              distinction="이용률(%)은 이용 여부 비율, 이용정도는 빈도/점수 척도")
    _add_edge(G, degree_id, rate_id, "CONFUSED_WITH",
              rule_id="domain_aug",
              distinction="이용정도(빈도/점수)는 이용률(%)과 다른 지표")
    # 이용정도/빈도 동의어
    for syn in ["이용 빈도", "이용빈도"]:
        sid = _add_node(G, "Term", syn)
        _add_edge(G, sid, degree_id, "SYNONYM_OF")
        _add_edge(G, degree_id, sid, "SYNONYM_OF")

    # (3) 과다이용 인식 ↔ 과의존위험군 비율 혼동 (원본 anchor/avoid 처리)
    overuse_id = _add_node(G, "Term", "과다이용 인식")
    riskgroup_id = _add_node(G, "Term", "과의존위험군")
    _add_edge(G, overuse_id, riskgroup_id, "CONFUSED_WITH",
              rule_id="domain_aug",
              distinction="과다이용 '인식'(주관 평가)과 과의존위험군 '비율'(척도 분류)은 별개")
    _add_edge(G, riskgroup_id, overuse_id, "CONFUSED_WITH",
              rule_id="domain_aug",
              distinction="과의존위험군 비율은 척도 기반 분류이며 주관적 과다이용 인식과 다름")

    # (4) 교차분석 금지 보강 (H02/H03 기반의 실질 금지쌍)
    #     고위험군/잠재적위험군 세분화는 '전체(B2)' 기준에서만 존재.
    #     특정 대상(유아동/청소년/성인/60대) 내에서의 세분화는 통계표에 없음 → 금지.
    high_risk_id = _add_node(G, "Term", "고위험군")
    latent_id = _add_node(G, "Term", "잠재적위험군")
    for tg in ["유아동", "청소년", "성인", "60대"]:
        tg_term = _add_node(G, "Term", tg)
        for risk_term in [high_risk_id, latent_id]:
            _add_edge(G, tg_term, risk_term, "PROHIBITS_CROSS",
                      rule_id="rule_H02_과의존수준_하위분류_범위",
                      raw=f"{tg} × {G.nodes[risk_term]['label']}")
            _add_edge(G, risk_term, tg_term, "PROHIBITS_CROSS",
                      rule_id="rule_H02_과의존수준_하위분류_범위",
                      raw=f"{tg} × {G.nodes[risk_term]['label']}")


def _split_combo(combo: str) -> list:
    """
    '성별 × 연령대별 (예: ...)' / '유아동 x 위험군' / '유아동, 위험군' 등
    다양한 구분자로 표현된 교차 조합 문자열을 토큰 리스트로 분리한다.
    괄호 안 예시 설명은 제거한 뒤 축 이름만 추출한다.
    """
    if not isinstance(combo, str):
        return []
    # 1) 괄호 예시 제거: "성별 × 연령대별 (예: ...)" → "성별 × 연령대별"
    cleaned = re.sub(r"[\(\（\[][^\)\）\]]*[\)\）\]]", "", combo).strip()
    # 2) '기타 모든 ...' 같은 일반 서술은 교차쌍이 아니므로 스킵
    if cleaned.startswith("기타") or "이상 교차" in cleaned:
        return []
    # 3) 구분자로 분해
    for sep in ["×", "x", "X", "*", "vs", "VS", "+"]:
        if sep in cleaned:
            return [p.strip() for p in cleaned.split(sep) if p.strip()]
    # 콤마는 마지막에 시도 (축 이름 자체에 콤마가 드묾)
    if "," in cleaned:
        return [p.strip() for p in cleaned.split(",") if p.strip()]
    return [cleaned] if cleaned else []
    """
    '성별 × 연령대별 (예: ...)' / '유아동 x 위험군' / '유아동, 위험군' 등
    다양한 구분자로 표현된 교차 조합 문자열을 토큰 리스트로 분리한다.
    괄호 안 예시 설명은 제거한 뒤 축 이름만 추출한다.
    """
    if not isinstance(combo, str):
        return []
    # 1) 괄호 예시 제거: "성별 × 연령대별 (예: ...)" → "성별 × 연령대별"
    cleaned = re.sub(r"[\(\（\[][^\)\）\]]*[\)\）\]]", "", combo).strip()
    # 2) '기타 모든 ...' 같은 일반 서술은 교차쌍이 아니므로 스킵
    if cleaned.startswith("기타") or "이상 교차" in cleaned:
        return []
    # 3) 구분자로 분해
    for sep in ["×", "x", "X", "*", "vs", "VS", "+"]:
        if sep in cleaned:
            return [p.strip() for p in cleaned.split(sep) if p.strip()]
    # 콤마는 마지막에 시도 (축 이름 자체에 콤마가 드묾)
    if "," in cleaned:
        return [p.strip() for p in cleaned.split(",") if p.strip()]
    return [cleaned] if cleaned else []


# ---------------------------------------------------------
# 직접 실행 시: 그래프 통계 출력 (개발용)
# ---------------------------------------------------------
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "rag_retrieval_dictionary.json"
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    g = build_ontology_graph(d)

    print(f"노드 {g.number_of_nodes()}개 / 엣지 {g.number_of_edges()}개")
    # 타입별 노드 수 집계
    from collections import Counter
    ntypes = Counter(data.get("ntype") for _, data in g.nodes(data=True))
    etypes = Counter(data.get("etype") for _, _, data in g.edges(data=True))
    print("노드 타입:", dict(ntypes))
    print("엣지 타입:", dict(etypes))
