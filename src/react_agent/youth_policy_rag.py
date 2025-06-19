"""
LangGraph Studio용 청년정책 RAG 시스템
주거 관련 정책과 일자리 관련 정책에 대해서만 답변하고, 
그 외 질문에 대해서는 답변을 거부하는 시스템
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Literal, Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# LangChain imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# LangChain SQL imports
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI


# 환경변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryClassification(BaseModel):
    """질의 분류를 위한 구조화된 출력 모델"""
    lclsf_nm: Literal["주거", "일자리", "기타"] = Field(
        description="대분류(lclsf_nm): 주거, 일자리, 기타"
    )
    mclsf_nm: Optional[Literal[
        "대출, 이자, 전월세 등 금융지원",
        "임대주택, 기숙사 등 주거지원", 
        "이사비, 부동산 중개비 등 보조금지원",
        "전문인력양성, 훈련",
        "창업",
        "취업 전후 지원"
    ]] = Field(
        default=None,
        description="중분류(mclsf_nm): 주거-금융지원/주거지원/보조금지원, 일자리-훈련/창업/취업지원, 기타-없음"
    )
    confidence: float = Field(
        description="분류 신뢰도 (0.0-1.0)", 
        ge=0.0, 
        le=1.0
    )
    reasoning: str = Field(
        description="분류 근거 설명"
    )


class UserConditions(BaseModel):
    """사용자 조건 추출을 위한 구조화된 출력 모델"""
    age: Optional[int] = Field(
        default=None,
        description="사용자 나이"
    )
    mrg_stts_cd: Optional[Literal["기혼", "미혼"]] = Field(
        default=None,
        description="결혼 상태"
    )
    plcy_major_cd: Optional[Literal["인문계열", "자연계열", "사회계열", "상경계열", "이학계열", "공학계열", "예체능계열", "농산업계열"]] = Field(
        default=None,
        description="전공 계열"
    )
    job_cd: Optional[Literal["재직자", "미취업자", "자영업자", "(예비)창업자", "영농종사자", "비정규직"]] = Field(
        default=None,
        description="취업 상태"
    )
    school_cd: Optional[Literal["고졸 미만", "고교 재학", "고졸 예정", "고교 졸업", "대학 재학", "대졸 예정", "대학 졸업", "석·박사"]] = Field(
        default=None,
        description="학력 상태"
    )
    zip_cd: Optional[str] = Field(
        default=None,
        description="거주지 (광역시/도, 시군구)"
    )
    earn_etc_cn: Optional[str] = Field(
        default=None,
        description="소득 요건 (예: 중위소득 150% 이하, 월소득 200만원 이하 등)"
    )
    additional_requirement: Optional[str] = Field(
        default=None,
        description="기타 추가 요건이나 상황"
    )
    confidence: float = Field(
        description="조건 추출 신뢰도 (0.0-1.0)",
        ge=0.0,
        le=1.0
    )


class GraphState(TypedDict):
    """그래프 상태 정의"""
    messages: Annotated[List[BaseMessage], add_messages]  # LangGraph Studio 호환성을 위한 메시지 리스트
    query: str  # 사용자 질의
    classification: Optional[QueryClassification]  # 질의 분류 결과
    user_conditions: Optional[UserConditions]  # 사용자 조건 추출 결과
    generated_sql: Optional[str]  # 정책 검색을 위한 필터 쿼리
    sql_result: Optional[str]
    final_response: Optional[str]  # 최종 답변
    error: Optional[str]  # 오류 메시지
    timestamp: str  # 처리 시각


class YouthPolicyRAGConfig:
    """RAG 시스템 설정"""
    def __init__(self):
        # 데이터베이스 설정
        self.db_config = {
            'host': os.getenv("DB_HOST", 'localhost'),
            'database': os.getenv("DB_NAME", 'youth_policy'),
            'user': os.getenv("DB_USER", 'postgres'),
            'password': os.getenv("DB_PASSWORD", 'your_password'),
            'port': os.getenv("DB_PORT", 5432)
        }
        
        # OpenAI 설정
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")
        
        # RAG 설정
        self.top_k = int(os.getenv('TOP_K', 10))
        self.confidence_threshold = os.getenv('CONFIDENCE_THRESHOLD', 0.5)  # 분류 신뢰도 임계값
        
        # PostgreSQL URI 생성
        self.db_uri = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"

        # LangChain LLM 설정
        try:
            self.langchain_llm = ChatOpenAI(
                api_key=self.openai_api_key,
                temperature=0,
                verbose=True,
                model="gpt-4o"
            )
            logger.info("LangChain LLM 초기화 완료")
        except Exception as e:
            logger.warning(f"LangChain LLM 초기화 실패: {e}")
            self.langchain_llm = None
        # SQLDatabase 설정
        try:
            self.sql_database = SQLDatabase.from_uri(self.db_uri)
            logger.info("SQLDatabase 초기화 완료")
        except Exception as e:
            logger.warning(f"SQLDatabase 초기화 실패: {e}")
            self.sql_database = None
        
        # LangChain ChatOpenAI 모델 설정 (질의 분류용)
        self.chat_model = ChatOpenAI(
            api_key=self.openai_api_key,
            model="o3-mini"
        )


# 전역 설정 인스턴스
config = YouthPolicyRAGConfig()


def classify_query_node(state: GraphState) -> GraphState:
    """질의 분류 노드 - LangChain structured output 사용"""
    try:
        logger.info("질의 분류 시작")
        
        # 메시지에서 마지막 사용자 메시지 추출
        user_message = None
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                user_message = message.content
                break
        
        if not user_message:
            raise ValueError("사용자 메시지를 찾을 수 없습니다.")
        
        # 프롬프트 템플릿 정의
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 청년정책 질의 분류 전문가입니다. 
사용자의 질문을 다음과 같이 대분류(lclsf_nm)와 중분류(mclsf_nm)로 분류해주세요:

**대분류 (lclsf_nm):**
- '주거': 주거 관련 정책
- '일자리': 일자리 관련 정책  
- '기타': 그 외 모든 질문

**중분류 (mclsf_nm):**

**lclsf_nm이 '주거'일 때:**
- "대출, 이자, 전월세 등 금융지원": 전세자금 대출, 임차보증금 대출, 대출이자 지원, 월세 대출 등
- "임대주택, 기숙사 등 주거지원": 청년 임대주택, 기숙사, 주거공간 제공, 공공임대 등
- "이사비, 부동산 중개비 등 보조금지원: 중개수수료 지원, 이사비 지원, 월세 보조금, 주거급여 등

**lclsf_nm이 '일자리'일 때:**
- '전문인력양성, 훈련': 직업 훈련, 기술 교육, 자격증 취득 지원, 역량강화, 교육과정 등
- '창업': 창업 지원, 창업자금, 창업 교육, 창업보육센터, 사업자 등록 등
- '취업 전후 지원': 취업 지원, 구직활동 지원, 취업 후 정착 지원, 인턴십, 채용박람회 등

**lclsf_nm이 '기타'일 때:**
- 중분류 없음

주거와 일자리 관련 키워드를 정확히 식별하고, 애매한 경우에는 기타로 분류하세요."""),
            ("human", "다음 질문을 분류해주세요: {query}")
        ])
        
        # 구조화된 출력을 위한 체인 생성
        structured_llm = config.chat_model.with_structured_output(QueryClassification)
        chain = prompt | structured_llm
        
        # 분류 실행
        classification = chain.invoke({"query": user_message})
        
        logger.info(f"질의 분류 완료: {classification.lclsf_nm}/{classification.mclsf_nm} (신뢰도: {classification.confidence})")
        
        return {
            **state,
            "query": user_message,
            "classification": classification
        }
        
    except Exception as e:
        logger.error(f"질의 분류 실패: {e}")
        return {
            **state,
            "error": f"질의 분류 실패: {str(e)}"
        }


def route_after_classification(state: GraphState) -> Literal["continue", "reject"]:
    """분류 결과에 따른 라우팅 결정"""
    if state.get("error"):
        return "reject"
    
    classification = state.get("classification")
    if not classification:
        return "reject"
    # 주거 또는 일자리 관련이고 신뢰도가 임계값 이상인 경우만 계속 진행
    if classification.lclsf_nm in ["주거", "일자리"]:
        logger.info(f"질의 승인: {classification.lclsf_nm} (신뢰도: {classification.confidence})")
        return "continue"
    else:
        logger.info(f"질의 거부: {classification.lclsf_nm} (신뢰도: {classification.confidence})")
        return "reject"


def extract_user_conditions_node(state: GraphState) -> GraphState:
    """사용자 조건 추출 노드 - LangChain structured output 사용"""
    try:
        logger.info("사용자 조건 추출 시작")
        
        # 프롬프트 템플릿 정의
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 사용자의 질문에서 개인 조건을 추출하는 전문가입니다.
사용자의 질문을 분석하여 다음 조건들을 추출해주세요:

**추출할 조건들:**
1. age: 나이 (숫자로)
2. mrg_stts_cd: 결혼 상태 ('기혼', '미혼' 중 하나)
3. plcy_major_cd: 전공 계열 ('인문계열', '자연계열', '사회계열', '상경계열', '이학계열', '공학계열', '예체능계열', '농산업계열' 중 하나)
4. job_cd: 취업 상태 ('재직자', '미취업자', '자영업자', '(예비)창업자', '영농종사자', '비정규직' 중 하나)
5. school_cd: 학력 상태 ('고졸 미만', '고교 재학', '고졸 예정', '고교 졸업', '대학 재학', '대졸 예정', '대학 졸업', '석·박사' 중 하나)
6. zip_cd: 거주지 (광역지자체, 기초지자체 형태로)
7. earn_etc_cn: 소득 요건 (구체적인 소득 수준이나 조건)
8. additional_requirement: 기타 추가 요건이나 특별한 상황

**추출 규칙:**
- 명시적으로 언급되지 않은 조건은 None으로 설정
- 추론이나 가정하지 말고, 명확히 언급된 내용만 추출
- 거주지는 "서울특별시", "대구광역시", "경상북도", "전북특별자치도", "서울 구로구", "대구 달서구", "경기도 수원시 팔달구" 의 형태로 추출
- 소득은 "월소득 200만원 이하", "중위소득 150% 이하" 등의 형태로 추출
- 신뢰도는 추출된 정보의 명확성과 완성도를 기준으로 평가"""),
            ("human", "다음 질문에서 사용자의 개인 조건을 추출해주세요: {query}")
        ])
        
        # 구조화된 출력을 위한 체인 생성
        structured_llm = config.chat_model.with_structured_output(UserConditions)
        chain = prompt | structured_llm
        
        # 조건 추출 실행
        user_conditions = chain.invoke({"query": state['query']})
        
        logger.info(f"사용자 조건 추출 완료 (신뢰도: {user_conditions.confidence})")
        logger.info(f"추출된 조건: 나이={user_conditions.age}, 결혼상태={user_conditions.mrg_stts_cd}, 거주지={user_conditions.zip_cd}")
        
        return {
            **state,
            "user_conditions": user_conditions
        }
        
    except Exception as e:
        logger.error(f"사용자 조건 추출 실패: {e}")
        return {
            **state,
            "error": f"사용자 조건 추출 실패: {str(e)}"
        }


def generate_sql_response_node(state: GraphState) -> GraphState:
    """SQLDatabaseChain을 사용한 정책 검색 및 응답 생성 노드"""
    try:
        logger.info("SQLDatabaseChain을 사용한 정책 검색 및 응답 생성 시작")
        
        classification = state["classification"]
        user_conditions = state.get("user_conditions")
        query = state["query"]
        
        # SQLDatabaseChain이 사용 가능한지 확인
        if config.sql_database is None or config.langchain_llm is None:
            logger.error("SQLDatabaseChain을 사용할 수 없습니다. 데이터베이스 또는 LLM이 설정되지 않았습니다.")
            error_message = "정책 검색을 위한 데이터베이스 연결이 설정되지 않았습니다."
            ai_message = AIMessage(content=error_message)
            return {
                **state,
                "messages": state["messages"] + [ai_message],
                "error": error_message
            }
        
        # 사용자 조건을 바탕으로 질의 강화
        enhanced_query = build_enhanced_query(query, classification, user_conditions)
        
        # 커스텀 프롬프트 생성
        custom_prompt = create_youth_policy_prompt(classification.lclsf_nm)
        
        try:
            # SQLDatabaseChain 생성
            db_chain = SQLDatabaseChain.from_llm(
                llm=config.langchain_llm,
                db=config.sql_database,
                verbose=True,
                use_query_checker=True,
                return_intermediate_steps=True,
                prompt=custom_prompt
            )
            
            # 질의 실행
            result = db_chain(enhanced_query)
            
            logger.info("SQLDatabaseChain 응답 생성 완료")
            
        except Exception as e:
            logger.error(f"SQLDatabaseChain 실행 실패: {e}")
            # SQL 실행 실패 시 오류 메시지 반환
            error_message = f"정책 검색 중 오류가 발생했습니다: {str(e)}"
            ai_message = AIMessage(content=error_message)
            return {
                **state,
                "messages": state["messages"] + [ai_message],
                "error": error_message
            }
        
        # 메시지 리스트에 AI 응답 추가
        ai_message = AIMessage(content=result["answer"])
        
        return {
            **state,
            "messages": state["messages"] + [ai_message],
            "final_response": result["answer"],
            "generated_sql": result["intermediate_steps"][0]["query"],
            "sql_result": result["intermediate_steps"][0]["result"],
        }
        
    except Exception as e:
        logger.error(f"SQL 응답 생성 실패: {e}")
        error_message = f"정책 검색 중 오류가 발생했습니다: {str(e)}"
        ai_message = AIMessage(content=error_message)
        
        return {
            **state,
            "messages": state["messages"] + [ai_message],
            "error": error_message
        }


def clean_sql_query(sql_query: str) -> str:
    """SQL 쿼리에서 markdown 코드 블록 마커를 제거"""
    import re
    
    # ```sql과 ``` 마커 제거
    cleaned = re.sub(r'```sql\s*', '', sql_query)
    cleaned = re.sub(r'```\s*$', '', cleaned)
    cleaned = re.sub(r'```', '', cleaned)
    
    # 앞뒤 공백 제거
    cleaned = cleaned.strip()
    
    return cleaned


def build_enhanced_query(original_query: str, classification: Query-Classification, user_conditions: Optional[UserConditions]) -> str:
    """사용자 조건을 바탕으로 질의를 강화"""
    enhanced_parts = [f"사용자 질문: {original_query}"]
    
    # 분류 정보 추가
    enhanced_parts.append(f"정책대분류명: {classification.lclsf_nm}")
    if classification.mclsf_nm:
        enhanced_parts.append(f"정책중분류명: {classification.mclsf_nm}")
    
    # 사용자 조건 추가
    if user_conditions:
        condition_parts = []
        if user_conditions.age:
            condition_parts.append(f"나이: {user_conditions.age}세")
        if user_conditions.mrg_stts_cd:
            condition_parts.append(f"결혼상태: {user_conditions.mrg_stts_cd}")
        if user_conditions.job_cd:
            condition_parts.append(f"취업상태: {user_conditions.job_cd}")
        if user_conditions.school_cd:
            condition_parts.append(f"학력: {user_conditions.school_cd}")
        if user_conditions.zip_cd:
            condition_parts.append(f"거주지: {user_conditions.zip_cd}")
        if user_conditions.earn_etc_cn:
            condition_parts.append(f"소득조건: {user_conditions.earn_etc_cn}")
        
        if condition_parts:
            enhanced_parts.append(f"사용자 조건: {', '.join(condition_parts)}")
    
    return "\n".join(enhanced_parts)


def create_youth_policy_prompt(category: str):
    """청년정책 특화 프롬프트 생성"""
    category_desc = "주거 관련" if category == "주거" else "일자리 관련"
    
    template = f"""당신은 청년정책 전문 상담사입니다. 
주어진 데이터베이스에서 {category_desc} 정책을 검색하여 사용자의 질문에 답변해주세요.

데이터베이스 테이블 정보:
- policies: 정책 기본 정보 (정책명, 설명, 지원내용, 신청방법 등)
- policy_conditions: 정책 지원 조건 (나이, 소득, 학력, 거주지 등)

다음 조건으로 검색해주세요:
1. lclsf_nm (대분류)가 '{category}'인 정책만 검색
2. 사용자 조건(나이, 거주지 등)에 맞는 정책 우선 검색
3. 최대 {config.top_k}개의 정책 반환

{{table_info}}

Question: {{input}}
SQLQuery: 적절한 SQL 쿼리를 생성하세요
SQLResult: 쿼리 결과
Answer: 위 결과를 바탕으로 정확하고 도움이 되는 답변을 제공하세요"""

    return PromptTemplate(
        input_variables=["table_info", "input"],
        template=template
    )


def reject_query_node(state: GraphState) -> GraphState:
    """질의 거부 노드"""
    logger.info("질의 거부 처리")
    
    classification = state.get("classification")
    
    if state.get("error"):
        response = f"""죄송합니다. 질문을 처리하는 중 오류가 발생했습니다.

오류: {state['error']}

다시 시도해 주시기 바랍니다."""
    else:
        response = f"""죄송합니다. 저는 청년들의 **주거 관련 정책**과 **일자리 관련 정책**에 대해서만 도움을 드릴 수 있습니다.

현재 질문은 '{classification.lclsf_nm if classification else '분류불가'}' 카테고리로 분류되어 답변을 드릴 수 없습니다.

다음과 같은 질문에 대해 도움을 드릴 수 있습니다:

**🏠 주거 관련 정책:**
- 임대료 지원, 주택 구입 지원
- 중개수수료 지원, 전세자금 대출
- 주거급여, 청년 임대주택 등

**💼 일자리 관련 정책:**
- 취업 지원, 창업 지원
- 직업 훈련, 인턴십 프로그램
- 취업 수당, 고용보험 등

주거나 일자리와 관련된 질문으로 다시 문의해 주시면 더 나은 도움을 드리겠습니다."""
    
    # 메시지 리스트에 AI 응답 추가
    ai_message = AIMessage(content=response)
    
    return {
        **state,
        "messages": state["messages"] + [ai_message],
        "final_response": response
    }


def build_graph() -> StateGraph:
    """LangGraph 워크플로우 구축"""
    # StateGraph 생성
    builder = StateGraph(GraphState)
    
    # 노드 추가
    builder.add_node("classify_query", classify_query_node)
    builder.add_node("extract_user_conditions", extract_user_conditions_node)
    builder.add_node("generate_sql_response", generate_sql_response_node)
    builder.add_node("reject_query", reject_query_node)
    
    # 엣지 정의
    builder.add_edge(START, "classify_query")
    # 조건부 엣지: 분류 결과에 따라 라우팅
    builder.add_conditional_edges(
        "classify_query",
        route_after_classification,
        {
            "continue": "extract_user_conditions",
            "reject": "reject_query"
        }
    )
    builder.add_edge("extract_user_conditions", "generate_sql_response")
    builder.add_edge("generate_sql_response", END)
    builder.add_edge("reject_query", END)
    
    return builder.compile()


# LangGraph Studio에서 사용할 그래프 인스턴스
graph = build_graph()
