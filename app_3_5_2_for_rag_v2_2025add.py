# =========================================================
# Streamlit 기반 스마트폰 과의존 실태조사 RAG 챗봇 v5.2
# 
# LangGraph 로직은 script/smart_langgraph.py에서 import
# =========================================================

from __future__ import annotations
import streamlit as st
import os
import sys
import pandas as pd
from textwrap import dedent

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

# =========================================================
# script 폴더를 Python path에 추가
# =========================================================
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "script")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# =========================================================
# LangGraph 모듈 import
# =========================================================
from smart_langgraph_for_3_5_2_v3_2025add import (
    # 상수
    YEAR_TO_FILENAME,
    BOT_IDENTITY,
    # RAG Dictionary 함수
    load_rag_dict,
    build_rag_dict_index,
    # 노드 팩토리 및 그래프 빌드
    create_node_functions,
    build_graph,
    # [KG 추가] 지식 그래프 빌더
    build_knowledge_graph,
)

# =========================================================
# Hugging Face 설정
# =========================================================
HF_REPO_ID = "Rosaldowithbaek/smartphoe_overdependence_survey_chromadb"
LOCAL_DB_PATH = "./chroma_db_store"
RAG_DICT_PATH = 'rag_retrieval_dictionary.json'

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="스마트폰 과의존 실태조사 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 커스텀 CSS - 포멀한 디자인
# =========================================================
st.markdown("""
<style>
    /* 전체 배경 및 폰트 */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* 헤더 스타일 */
    h1 {
        color: #1e3a5f;
        font-weight: 600;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 0.5rem;
    }
    
    /* 가이드 박스 - 포멀한 네이비/그레이 계열 */
    .guide-box {
        background: #1e3a5f;
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #0d2137;
    }
    
    .guide-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #ffffff;
        letter-spacing: 0.5px;
    }
    
    .guide-item {
        background: rgba(255,255,255,0.1);
        padding: 0.8rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        line-height: 1.6;
        border-left: 3px solid rgba(255,255,255,0.3);
    }
    
    .guide-item:last-child {
        margin-bottom: 0;
    }
    
    .guide-item strong {
        color: #a8c5e2;
    }
    
    .guide-item a {
        color: #7eb8e7;
        text-decoration: underline;
    }
    
    /* 상태 박스 */
    .status-box {
        background-color: #f0f4f8;
        padding: 0.8rem 1rem;
        border-radius: 6px;
        border-left: 4px solid #1e3a5f;
        margin: 0.5rem 0;
        font-weight: 500;
        color: #1e3a5f;
    }
    
    /* 검증 상태 배지 */
    .validation-pass {
        color: #1b5e20;
        font-weight: 600;
    }
    
    .validation-fail {
        color: #b71c1c;
        font-weight: 600;
    }
    
    .retry-badge {
        background-color: #fff3e0;
        color: #e65100;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* 사이드바 스타일 */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* 다운로드 섹션 */
    .download-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #dee2e6;
        margin-top: 1rem;
    }
    
    .download-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #1e3a5f;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 세션 상태 초기화
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "clarification_context" not in st.session_state:
    st.session_state.clarification_context = None

if "rag_dict" not in st.session_state:
    st.session_state.rag_dict = load_rag_dict(RAG_DICT_PATH)

if "rag_dict_index" not in st.session_state:
    st.session_state.rag_dict_index = build_rag_dict_index(st.session_state.rag_dict)

# [KG 추가] 지식 그래프를 세션 단위로 1회 빌드해 캐싱.
#           build_knowledge_graph 가 None(모듈 로드 실패)이면 KG 없이 동작.
if "knowledge_graph" not in st.session_state:
    if build_knowledge_graph is not None:
        try:
            st.session_state.knowledge_graph = build_knowledge_graph(
                st.session_state.rag_dict
            )
        except Exception as _kg_e:
            st.session_state.knowledge_graph = None
    else:
        st.session_state.knowledge_graph = None


# =========================================================
# Hugging Face에서 DB 다운로드
# =========================================================
@st.cache_resource
def download_chroma_db():
    """Hugging Face에서 Chroma DB를 다운로드한다."""
    if os.path.exists(LOCAL_DB_PATH) and os.listdir(LOCAL_DB_PATH):
        return LOCAL_DB_PATH, None
    
    try:
        from huggingface_hub import snapshot_download
        import stat
        
        downloaded_path = snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_DB_PATH,
            local_dir_use_symlinks=False
        )
        
        # 다운로드된 파일에 쓰기 권한 부여 (readonly DB 오류 방지)
        for root, dirs, files in os.walk(downloaded_path):
            for f in files:
                fp = os.path.join(root, f)
                os.chmod(fp, stat.S_IREAD | stat.S_IWRITE)
        
        return downloaded_path, None
    except Exception as e:
        return None, str(e)


# =========================================================
# 초기화 함수
# =========================================================
@st.cache_resource
def init_resources():
    """리소스(벡터스토어, LLM)를 초기화한다."""
    # API 키 확인
    api_key = None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY")
    except:
        pass
    
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        return None, None, "OpenAI API 키가 설정되지 않았습니다."
    
    os.environ['OPENAI_API_KEY'] = api_key
    
    # DB 경로 확인
    if not os.path.exists(LOCAL_DB_PATH):
        return None, None, f"Chroma DB를 찾을 수 없습니다: {LOCAL_DB_PATH}"
    
    try:
        # 임베딩 및 벡터스토어 초기화
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        vectorstore = Chroma(
            persist_directory=LOCAL_DB_PATH,
            embedding_function=embedding,
            collection_name="pdf_pages_with_summary_v2"
        )
        
        # LLM 설정 - 원본과 동일한 모델명 사용
        llms = {
            "router": ChatOpenAI(model="gpt-5-mini"),
            "chat_refer": ChatOpenAI(model="gpt-5-mini"),
            "parse_year": ChatOpenAI(model="gpt-5-mini"),
            "followup": ChatOpenAI(model="gpt-5-mini"),
            "casual": ChatOpenAI(model="gpt-5-mini"),
            "main": ChatOpenAI(model="gpt-5-mini"),
            "rewrite": ChatOpenAI(model="gpt-5-mini"),
            "validator": ChatOpenAI(model="gpt-5-mini"),
        }
        
        return vectorstore, llms, None
    except Exception as e:
        return None, None, str(e)


# =========================================================
# 테이블 파싱 및 렌더링
# =========================================================
def parse_markdown_table(text: str):
    """마크다운 테이블을 파싱한다."""
    tables = []
    lines = text.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('|') and line.endswith('|'):
            table_lines = []
            start_idx = i
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('|') and line.endswith('|'):
                    table_lines.append(line)
                    i += 1
                elif line.startswith('|---') or line.startswith('| ---'):
                    i += 1
                    continue
                else:
                    break
            
            if len(table_lines) >= 2:
                header_line = table_lines[0]
                headers = [h.strip() for h in header_line.split('|')[1:-1]]
                data_rows = []
                for row_line in table_lines[1:]:
                    if '---' in row_line:
                        continue
                    cells = [c.strip() for c in row_line.split('|')[1:-1]]
                    if len(cells) == len(headers):
                        data_rows.append(cells)
                
                if headers and data_rows:
                    tables.append({
                        'headers': headers,
                        'rows': data_rows,
                        'start_idx': start_idx,
                        'end_idx': i
                    })
        else:
            i += 1
    return tables


def render_answer_with_tables(answer: str) -> None:
    """테이블이 포함된 답변을 렌더링한다."""
    tables = parse_markdown_table(answer)
    if not tables:
        st.markdown(answer)
        return
    
    lines = answer.split('\n')
    current_pos = 0
    
    for table in tables:
        # 테이블 이전 텍스트 렌더링
        before_text = '\n'.join(lines[current_pos:table['start_idx']])
        if before_text.strip():
            st.markdown(before_text)
        
        # 테이블 렌더링
        try:
            df = pd.DataFrame(table['rows'], columns=table['headers'])
            st.dataframe(df, use_container_width=True, hide_index=True)
        except:
            st.markdown("| " + " | ".join(table['headers']) + " |")
            for row in table['rows']:
                st.markdown("| " + " | ".join(row) + " |")
        
        current_pos = table['end_idx']
    
    # 테이블 이후 텍스트 렌더링
    if current_pos < len(lines):
        after_text = '\n'.join(lines[current_pos:])
        if after_text.strip():
            st.markdown(after_text)


# =========================================================
# 상태 업데이트 콜백 생성
# =========================================================
def create_status_callback(status_placeholder):
    """
    Streamlit status_placeholder를 업데이트하는 콜백 함수를 생성한다.
    
    Args:
        status_placeholder: st.empty()로 생성된 placeholder
        
    Returns:
        callable: 상태 텍스트를 받아 placeholder를 업데이트하는 함수
    """
    def callback(status_text: str):
        status_placeholder.markdown(
            f'<div class="status-box">{status_text}</div>', 
            unsafe_allow_html=True
        )
    return callback

# 사용자 가이드 박스 (들여쓰기/개행 때문에 코드블록으로 보이는 문제 방지)
guide_html = """
<div class="guide-box">
  <div class="guide-title">📌 사용 안내</div>

  <div class="guide-item">
    <strong>용도:</strong> 스마트폰 과의존 실태조사 보고서(2020~2025) <strong>정보 검색용</strong>입니다. <br />
    인사이트 제공, 일반 대화, 보고서 외 정보 검색에는 적합하지 않습니다.
  </div>

  <div class="guide-item">
    <strong>검색 팁:</strong> 아래 3가지 요소를 포함하면 정확도가 높아집니다.<br />
    <br />
    <strong>① 연도</strong><br />
    • 2020~2025 중 선택 (미입력 시 2024~2025 적용)<br />
    • 💡 “최근 N년”은 기준연도 계산 후 2020~2025 밖 연도는 제외되어 범위가 좁아질 수 있어 숫자 연도 권장<br />
    <br />
    <strong>② 대상</strong><br />
    • 유아동(만3~9, 보호자응답) / 청소년(10~19) / 성인(20~59) / 60대(60~69, 고령층·시니어)<br />
    • ※ 70대 이상은 조사 대상 아님<br />
    <br />
    <strong>③ 지표</strong><br />
    • 콘텐츠 이용률(%) vs 콘텐츠 이용정도(빈도/점수) 구분해서 입력 등 구체적인 지표명<br />
    <br />
    <strong>TIP</strong><br />
    • 교차조건(성별/대상 등)이나 주제 키워드(숏폼/콘텐츠명/지표명/예방교육 등)를 추가하면 더 정확해집니다.<br />
    • 주제, 대상이 바뀌면 이전 대화 맥락이 오히려 정확한 답변에 방해가 되거나 검색이 안되는 결과로 나타날 수 있습니다. 초기화 이후 재검색하는 것을 권장합니다. <br />
    • 과도한 검색결과 방지를 위한 설정으로 인해 일부 연도가 검색 결과에서 누락될 수 있습니다. 그럴 때는 해당 연도를 지정해서 다시 질문해주세요.<br />
    • 보고서 내 유사한 내용이 다수 있어, 검색 성능이 안나올 수 있습니다. 요구하고자하는 바를 확실히 설명해주세요<br />
  </div>

  <div class="guide-item">
    <strong>주의:</strong> AI 답변에 <strong>오류(할루시네이션)</strong>가 있을 수 있습니다. <br />
    검색 결과를 바로 인용하지 마시고, <strong>원문을 통해 확인</strong>한 뒤 정보를 사용하십시오.<br />
    왼쪽의 pdf 보고서 다운로드 혹은
    <a href="https://www.nia.or.kr" target="_blank">NIA 홈페이지</a>에서 원문 확인 권장<br />
    단 pdf 다운로드 클릭시 기존 채팅이 멈출 수 있으니 사전에 혹은 모든 대화가 끝난 이후 다운로드를 해주세요<br />
  </div>
</div>
"""


# =========================================================
# 메인 UI
# =========================================================
def main():
    st.title("📊 스마트폰 과의존 실태조사 분석 시스템")

    # 사이드바
    with st.sidebar:
        st.header("시스템 정보")
        st.markdown(BOT_IDENTITY)

        st.divider()

        # PDF 다운로드 섹션
        st.subheader("📖 보고서 다운로드")
        for year, filename in YEAR_TO_FILENAME.items():
            pdf_path = f"data/{filename}"
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label=f"{year}년 보고서",
                        data=f,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )

        st.divider()

        # 대화 초기화 버튼
        if st.button("🔄 대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.clarification_context = None
            # 캐싱된 그래프/노드도 제거하여 다음 질문 시 재생성
            st.session_state.pop("graph", None)
            st.session_state.pop("node_functions", None)
            st.rerun()

        st.divider()

        # 디버그 모드 토글
        debug_mode = st.checkbox("디버그 모드", value=False)


    st.markdown(guide_html.strip(), unsafe_allow_html=True)

    # DB 다운로드
    if not os.path.exists(LOCAL_DB_PATH) or not os.listdir(LOCAL_DB_PATH):
        st.info("🔄 Chroma DB를 다운로드하고 있습니다...")
        with st.spinner(f"Hugging Face에서 다운로드 중... ({HF_REPO_ID})"):
            db_path, error = download_chroma_db()

        if error:
            st.error(f"DB 다운로드 실패: {error}")
            return
        else:
            st.success("DB 다운로드 완료!")
            st.rerun()

    # 리소스 초기화
    vectorstore, llms, error = init_resources()

    if error:
        st.error(f"초기화 오류: {error}")
        if "API" in error:
            st.info("Streamlit Secrets에 OPENAI_API_KEY를 설정해주세요.")
            with st.form("api_key_form"):
                api_key = st.text_input("OpenAI API 키", type="password")
                if st.form_submit_button("설정") and api_key:
                    os.environ['OPENAI_API_KEY'] = api_key
                    st.rerun()
        return

    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                render_answer_with_tables(message["content"])
            else:
                st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요... (예: 2025년 청소년 과의존률은?)"):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 어시스턴트 응답
        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()

            try:
                # 상태 콜백 생성
                status_callback = create_status_callback(status_placeholder)

                # 그래프를 세션 단위로 캐싱 (status_callback만 매번 갱신)
                if "graph" not in st.session_state or "node_functions" not in st.session_state:
                    st.session_state.node_functions = create_node_functions(
                        vectorstore,
                        llms,
                        status_callback,  # 초기 콜백
                        st.session_state.rag_dict_index,
                        knowledge_graph=st.session_state.get("knowledge_graph"),  # [KG 추가]
                        )
                    st.session_state.graph = build_graph(st.session_state.node_functions)



                # status_callback은 매 턴마다 갱신 필요 (placeholder가 달라지므로)
                st.session_state.node_functions["_status_callback_ref"][0] = status_callback

                config = {"configurable": {"thread_id": "streamlit_session","recursion_limit": 50}}

                # 그래프 실행
                result = st.session_state.graph.invoke(
                    {
                        "input": prompt,
                        "chat_history": st.session_state.chat_history,
                        "session_id": "streamlit_session",
                        "clarification_context": st.session_state.clarification_context,
                    },
                    config=config
                )

                # 상태 표시 제거
                status_placeholder.empty()

                # Clarification context 저장
                if result.get("clarification_context"):
                    st.session_state.clarification_context = result["clarification_context"]
                else:
                    st.session_state.clarification_context = None

                # 최종 답변 표시
                final_answer = result.get("final_answer", "답변을 생성하지 못했습니다.")

                with answer_placeholder.container():
                    render_answer_with_tables(final_answer)

                # 디버그 정보 표시
                if debug_mode:
                    with st.expander("🔍 디버그 정보", expanded=False):
                        # ========== 기본 정보 ==========
                        st.markdown("#### 📌 기본 정보")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.write(f"**Intent:** {result.get('intent', 'N/A')}")
                            st.write(f"**Followup Type:** {result.get('followup_type', 'N/A')}")
                            st.write(f"**Is Chat Reference:** {result.get('is_chat_reference', 'N/A')}")

                        with col2:
                            st.write(f"**Retry Count:** {result.get('retry_count', 0)}")
                            st.write(f"**Retry Type:** {result.get('retry_type', 'N/A')}")
                            st.write(f"**Default Years Used:** {result.get('used_default_years', False)}")

                        with col3:
                            validation_result = result.get('validation_result', 'N/A')
                            if validation_result == "PASS":
                                st.markdown(f"**Validation:** <span class='validation-pass'>{validation_result}</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"**Validation:** <span class='validation-fail'>{validation_result}</span>", unsafe_allow_html=True)
                            st.write(f"**Safety:** passed={result.get('safety_passed', 'N/A')}")

                        # ========== 검색 계획 ==========
                        st.markdown("---")
                        st.markdown("#### 🎯 검색 계획 (Plan)")
                        if result.get("plan"):
                            plan = result["plan"]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**검색 연도:** {plan.get('years', [])}")
                                st.write(f"**파일 필터:** {len(plan.get('file_name_filters', []))}개")
                                for fn in plan.get('file_name_filters', []):
                                    st.caption(f"  └ {fn}")
                            with col2:
                                st.write(f"**Resolved Question:**")
                                st.caption(plan.get('resolved_question', 'N/A')[:150])

                        # ========== 쿼리 정보 ==========
                        st.markdown("---")
                        st.markdown("#### 🔧 쿼리 정보")
                        if result.get("rewritten_queries"):
                            queries = result["rewritten_queries"]
                            st.write(f"**Rewritten Queries ({len(queries)}개):**")
                            for i, q in enumerate(queries, 1):
                                st.caption(f"  {i}. {q}")
                        
                        # 연도별 쿼리 매핑
                        debug_info = result.get("debug_info") or {}
                        if debug_info.get("year_query_map"):
                            st.write("**연도별 쿼리 매핑:**")
                            for year, year_queries in debug_info["year_query_map"].items():
                                st.caption(f"  {year}년: {len(year_queries)}개 쿼리")

                        # ========== 검색 결과 ==========
                        st.markdown("---")
                        st.markdown("#### 📚 검색 결과")
                        if result.get("retrieval"):
                            retrieval = result["retrieval"]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**총 문서 수:** {retrieval.get('doc_count', 0)}")
                                st.write(f"**Parent IDs:** {len(retrieval.get('parent_ids', []))}개")
                            with col2:
                                st.write(f"**검색된 파일:**")
                                for fn in retrieval.get('files_searched', []):
                                    st.caption(f"  └ {fn}")
                        
                        # ========== 재검색 원인 분석 ==========
                        if debug_info.get("extract_figures"):
                            st.markdown("---")
                            st.markdown("#### 🔄 재검색 원인 분석")
                            ef = debug_info["extract_figures"]
                            
                        # 연도별 추출 상태
                        if ef.get("year_extract_detail"):
                            st.write("**연도별 추출 상태:**")
                            for year, detail in ef["year_extract_detail"].items():
                                status_icon = "✅" if not detail.get("is_missing") else "❌"
                                docs = detail.get("docs_retrieved", 0)
                                reason = detail.get("extraction_status", "UNKNOWN")
                                st.caption(f"  {status_icon} {year}년: 문서 {docs}개 검색 → 추출상태: {reason}")
    
                        # 추가 검색 트리거 여부
                        if ef.get("supplemental_search_triggered"):
                            st.warning("**⚡ 추가 검색 트리거됨**")
                        if ef.get("supplemental_search_reason"):
                                st.json(ef["supplemental_search_reason"])
                                
                            # Missing 이유
                                if debug_info.get("missing_years_reason"):
                                    st.write("**Missing 판정 이유:**")
                                    for year, reason in debug_info["missing_years_reason"].items():
                                        st.caption(f"  - {year}년: {reason}")
                        
                        # 연도별 문서 분포 (핵심 디버그 정보)
                        if debug_info.get("year_doc_distribution"):
                            st.write("**⭐ 연도별 문서 분포:**")
                            dist = debug_info["year_doc_distribution"]
                            cols = st.columns(len(dist))
                            for i, (year, count) in enumerate(sorted(dist.items())):
                                with cols[i]:
                                    color = "🟢" if count >= 3 else "🟡" if count >= 1 else "🔴"
                                    st.metric(f"{year}년", f"{color} {count}개")

                        # ========== Dict Hint ==========
                        st.markdown("---")
                        st.markdown("#### 📖 Dict Hint")
                        if result.get("dict_hint"):
                            dh = result["dict_hint"]
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Topic Code:** {dh.get('topic_code', 'N/A')}")
                                st.write(f"**Target Group:** {dh.get('target_group', 'N/A')}")
                            with col2:
                                st.write(f"**Anchor Terms:** {dh.get('anchor_terms', [])}")
                                st.write(f"**Avoid Terms:** {dh.get('avoid_terms', [])}")
                                if dh.get("scope_warnings"):
                                    st.write(f"**⚠️ Scope Warnings:** {dh.get('scope_warnings', [])}")

                        # ========== Validation 상세 ==========
                        st.markdown("---")
                        st.markdown("#### ✅ Validation 상세")
                        col1, col2 = st.columns(2)
                        with col1:
                            if result.get("validation_reason"):
                                st.write(f"**Validation Reason:**")
                                st.caption(result['validation_reason'][:200])
                            
                            # Validator 오버라이드 정보
                            if debug_info.get("validator_override"):
                                override = debug_info["validator_override"]
                                st.warning(f"**⚡ Validator Override:**\n"
                                          f"원래 결과: {override.get('original', 'N/A')}\n"
                                          f"이유: {override.get('reason', 'N/A')}")
                        
                        with col2:
                            # Scope Warnings (debug_info에서)
                            if debug_info.get("scope_warnings"):
                                st.write("**Scope Warnings:**")
                                for warn in debug_info["scope_warnings"]:
                                    st.caption(f"  ⚠️ {warn}")
                            
                            if result.get("validator_output"):
                                vo = result["validator_output"]
                                if vo.get("corrected_answer"):
                                    st.write("**Corrected Answer 존재:** Yes")

                        # ========== 기타 디버그 정보 ==========
                        if debug_info:
                            other_keys = [k for k in debug_info.keys() 
                                         if k not in ["year_query_map","extract_figures","year_doc_distribution", 
                                                      "validator_override", "scope_warnings",
                                                      "sanitized_has_extracted"]]
                            if other_keys:
                                st.markdown("---")
                                st.markdown("#### 🔧 기타 디버그 정보")
                                for key in other_keys:
                                    val = debug_info[key]
                                    if isinstance(val, dict):
                                        st.write(f"**{key}:**")
                                        st.json(val)
                                    else:
                                        st.write(f"**{key}:** {val}")

                # 세션 상태 업데이트
                st.session_state.messages.append({"role": "assistant", "content": final_answer})
                st.session_state.chat_history.append(HumanMessage(content=prompt))
                st.session_state.chat_history.append(AIMessage(content=final_answer))

                # 히스토리 길이 제한
                if len(st.session_state.chat_history) > 20:
                    st.session_state.chat_history = st.session_state.chat_history[-20:]

            except Exception as e:
                status_placeholder.empty()
                st.error(f"오류가 발생했습니다: {str(e)}")
                if debug_mode:
                    import traceback
                    st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
