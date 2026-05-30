# =========================================================
# kg_reasoner.py
# ---------------------------------------------------------
# 역할:
#   통합 KG(온톨로지 + 사실 계층) 위에서 추론을 수행해
#   기존 infer_dict_hint() 가 하던 일을 "그래프 탐색"으로 대체/확장한다.
#
# 기존 방식의 한계:
#   - infer_dict_hint 는 `if "이용률" in q` 식 하드코딩 + 부분일치라
#     (a) 동의어/별칭이 사전에 if문으로 박혀 있어야만 작동
#     (b) "쇼츠 조절" 같은 다중 개념 결합 질의에서 토픽 추론이 약함
#     (c) 혼동쌍 회피어를 매번 손으로 나열
#
# KG 방식의 이점:
#   - 질의 토큰을 그래프 노드에 매칭한 뒤 엣지를 따라가며
#     동의어 → 핵심개념 → 토픽 → 배너까지 한 번에 확장
#   - CONFUSED_WITH 엣지로 회피어(avoid_terms)를 자동 산출
#   - PROHIBITS_CROSS 엣지로 교차분석 금지 여부를 구조적으로 판정
#
# 반환 형식은 기존 infer_dict_hint 와 호환되도록 유지하여
# 다운스트림 노드(plan_search 등)를 수정 없이 재사용한다.
# =========================================================

from __future__ import annotations
import logging
from typing import Optional, List, Dict, Any, Set

import networkx as nx

logger = logging.getLogger(__name__)


def _nid(ntype: str, label: str) -> str:
    return f"{ntype}:{str(label).strip()}"


class KGReasoner:
    """통합 지식그래프 기반 추론기."""

    def __init__(self, graph: nx.MultiDiGraph):
        """
        Args:
            graph: build_ontology_graph()로 만든 뒤 fact_layer로 보강한 KG
        """
        self.G = graph
        # 성능을 위해 Term 라벨 → 노드ID 룩업 테이블을 사전 구축
        self._term_index = self._build_term_index()

    # -----------------------------------------------------
    # 내부: 라벨 룩업 인덱스
    # -----------------------------------------------------
    def _build_term_index(self) -> Dict[str, str]:
        """
        Term/CoreConcept/Factor/TargetGroup 노드의 라벨을 키로 하는
        역인덱스를 만든다. (질의 토큰 매칭 가속)
        """
        index = {}
        matchable = {"Term", "CoreConcept", "Factor", "TargetGroup",
                     "Metric", "Subtopic"}
        for nid, data in self.G.nodes(data=True):
            if data.get("ntype") in matchable:
                label = data.get("label", "")
                if label:
                    index[label] = nid
        return index

    # -----------------------------------------------------
    # 1) 질의 → 매칭 노드 집합
    # -----------------------------------------------------
    def _match_nodes(self, query: str) -> List[str]:
        """
        질의 문자열에서 그래프 노드 라벨을 부분일치로 찾아낸다.
        긴 라벨 우선(더 구체적인 매칭 우선)으로 정렬.
        """
        q = query or ""
        hits = []
        # 긴 라벨부터 검사해 '과의존위험군'이 '과의존'보다 먼저 매칭되게 함
        for label in sorted(self._term_index.keys(), key=len, reverse=True):
            if label and label in q:
                hits.append(self._term_index[label])
        return hits

    # -----------------------------------------------------
    # 2) 동의어/핵심개념 확장 (SYNONYM_OF 따라가기)
    # -----------------------------------------------------
    def _expand_synonyms(self, node_ids: List[str], max_per_node: int = 3) -> List[str]:
        """
        매칭된 노드에서 SYNONYM_OF 엣지를 1홉 따라가 앵커 후보를 늘린다.
        """
        anchors: List[str] = []
        for nid in node_ids:
            # 자신이 핵심개념이면 라벨 자체를 앵커로
            data = self.G.nodes[nid]
            if data.get("ntype") in {"CoreConcept", "Factor", "Metric"}:
                anchors.append(data.get("label", ""))
            # SYNONYM_OF 로 연결된 이웃(핵심개념/요인 방향)
            cnt = 0
            for _, dst, edata in self.G.out_edges(nid, data=True):
                if edata.get("etype") == "SYNONYM_OF":
                    anchors.append(self.G.nodes[dst].get("label", ""))
                    cnt += 1
                    if cnt >= max_per_node:
                        break
        return anchors

    # -----------------------------------------------------
    # 3) 토픽 라우팅 (ROUTES_TO, 가중치 합산)
    # -----------------------------------------------------
    def _route_topic(self, node_ids: List[str]) -> Optional[str]:
        """
        매칭 노드들에서 ROUTES_TO 엣지를 따라가 토픽 코드를 투표 집계한다.
        primary(weight 1.0)가 secondary(0.4)보다 우선되도록 가중합.
        """
        scores: Dict[str, float] = {}
        for nid in node_ids:
            for _, dst, edata in self.G.out_edges(nid, data=True):
                if edata.get("etype") != "ROUTES_TO":
                    continue
                dst_data = self.G.nodes[dst]
                if dst_data.get("ntype") != "Topic":
                    continue
                code = dst_data.get("code", "")
                if not code:
                    continue
                scores[code] = scores.get(code, 0.0) + float(edata.get("weight", 0.5))
        if not scores:
            return None
        # 최고 점수 토픽 코드 반환 (동점 시 코드 사전순)
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[0][0]

    # -----------------------------------------------------
    # 4) 회피어 산출 (CONFUSED_WITH 따라가기)
    # -----------------------------------------------------
    def _avoid_terms(self, node_ids: List[str]) -> List[Dict[str, str]]:
        """
        매칭 노드와 혼동되는 상대어를 CONFUSED_WITH 엣지로 수집한다.
        각 회피어에 구분(distinction) 힌트를 함께 담아 반환.
        """
        avoid = []
        seen: Set[str] = set()
        for nid in node_ids:
            for _, dst, edata in self.G.out_edges(nid, data=True):
                if edata.get("etype") != "CONFUSED_WITH":
                    continue
                dst_label = self.G.nodes[dst].get("label", "")
                if dst_label and dst_label not in seen:
                    seen.add(dst_label)
                    avoid.append({
                        "term": dst_label,
                        "distinction": edata.get("distinction", ""),
                        "routing_hint": edata.get("routing_hint", ""),
                    })
        return avoid

    # -----------------------------------------------------
    # 5) 대상(TargetGroup) 해석 (별칭 ALIAS_OF 포함)
    # -----------------------------------------------------
    def _resolve_target(self, query: str, node_ids: List[str]) -> str:
        """
        질의에서 조사 대상을 식별한다. 별칭(시니어→60대 등)도 ALIAS_OF로 해석.
        """
        # 직접 매칭된 TargetGroup
        for nid in node_ids:
            if self.G.nodes[nid].get("ntype") == "TargetGroup":
                return self.G.nodes[nid].get("label", "")
        # 별칭 → ALIAS_OF
        for nid in node_ids:
            for _, dst, edata in self.G.out_edges(nid, data=True):
                if edata.get("etype") == "ALIAS_OF":
                    return self.G.nodes[dst].get("label", "")
        return ""

    # -----------------------------------------------------
    # 6) 교차분석 금지 판정 (PROHIBITS_CROSS)
    # -----------------------------------------------------
    def check_prohibited_cross(self, query: str) -> List[Dict[str, str]]:
        """
        질의에 등장한 토큰들이 서로 교차분석 금지 관계인지 검사한다.
        예: '유아동' + '과의존위험군' 동시 등장 → 금지 조합 경고.

        Returns:
            금지 조합 리스트 [{"a":.., "b":.., "rule_id":..}, ...]
        """
        node_ids = self._match_nodes(query)
        labels = {self.G.nodes[n].get("label", "") for n in node_ids}
        violations = []
        seen_pairs: Set[frozenset] = set()

        for nid in node_ids:
            for _, dst, edata in self.G.out_edges(nid, data=True):
                if edata.get("etype") != "PROHIBITS_CROSS":
                    continue
                dst_label = self.G.nodes[dst].get("label", "")
                src_label = self.G.nodes[nid].get("label", "")
                # 양쪽 토큰이 모두 질의에 등장할 때만 위반으로 판정
                if dst_label in labels:
                    pair = frozenset({src_label, dst_label})
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        violations.append({
                            "a": src_label, "b": dst_label,
                            "rule_id": edata.get("rule_id", ""),
                            "raw": edata.get("raw", ""),
                        })
        return violations

    # -----------------------------------------------------
    # 통합 진입점: 기존 infer_dict_hint 와 호환되는 dict 반환
    # -----------------------------------------------------
    def infer_hint(self, text: str, context_text: str = "") -> Dict[str, Any]:
        """
        그래프 추론으로 검색 힌트를 생성한다.
        기존 infer_dict_hint() 의 반환 스키마를 유지하되,
        KG 전용 필드(prohibited_cross, confused_pairs)를 추가한다.

        Returns(기존 호환 + 확장):
            {
              "is_rag_like": bool,
              "topic_code": str,        # 'T05' 등
              "target_group": str,
              "anchor_terms": List[str],
              "avoid_terms": List[str],
              "needs_appendix_table": bool,
              "scope_warnings": List[str],
              # --- KG 확장 필드 ---
              "prohibited_cross": List[dict],
              "confused_pairs": List[dict],
              "matched_nodes": List[str],
            }
        """
        combined = f"{text or ''} {context_text or ''}".strip()
        node_ids = self._match_nodes(combined)

        # 토픽 라우팅
        topic_code = self._route_topic(node_ids) or ""
        is_rag_like = bool(topic_code) or bool(node_ids)

        # 앵커어(동의어 확장 결과 + 매칭된 핵심 라벨)
        anchors = self._expand_synonyms(node_ids)
        for nid in node_ids:
            lbl = self.G.nodes[nid].get("label", "")
            if lbl:
                anchors.append(lbl)

        # 회피어 (혼동쌍)
        confused = self._avoid_terms(node_ids)
        avoid = [c["term"] for c in confused]

        # 대상
        target_group = self._resolve_target(text or "", node_ids)

        # 교차분석 금지 검사
        prohibited = self.check_prohibited_cross(combined)

        # scope_warnings 구성 (기존 포맷 계승)
        scope_warnings = []
        for c in confused:
            if c.get("distinction"):
                scope_warnings.append(
                    f"★ '{c['term']}'와 혼동 주의: {c['distinction']}"
                )
        for p in prohibited:
            scope_warnings.append(
                f"⛔ 교차분석 금지 조합 감지: {p['a']} × {p['b']} "
                f"(규칙 {p['rule_id']}). 보고서에 해당 교차표가 없을 수 있습니다."
            )

        # 부록 표 필요 여부: 교차 금지가 아닌 '대상 내 위험군 비교'면 부록 참조 신호
        needs_appendix_table = bool(prohibited)

        def _uniq(lst):
            seen, out = set(), []
            for x in lst:
                x = str(x).strip()
                if x and x not in seen:
                    seen.add(x)
                    out.append(x)
            return out

        return {
            "is_rag_like": is_rag_like,
            "topic_code": topic_code,
            "target_group": target_group,
            "anchor_terms": _uniq(anchors),
            "avoid_terms": _uniq(avoid),
            "needs_appendix_table": needs_appendix_table,
            "scope_warnings": scope_warnings,
            # KG 확장 필드
            "prohibited_cross": prohibited,
            "confused_pairs": confused,
            "matched_nodes": [self.G.nodes[n].get("label", "") for n in node_ids],
        }
