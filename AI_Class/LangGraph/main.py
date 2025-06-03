from llm_model import llm_model
from embeddings import chroma_vectorstore
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from typing import TypedDict, Annotated
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from rag_bot import tutorial_chain

# .env 파일에서 환경 변수 로드
load_dotenv()

# LLM Load
llm = llm_model()

# 그래프 상태를 정의하는 클래스
class MyState(TypedDict):
    question: Annotated[str, "question"]        # 질문
    context: Annotated[str, "context"]          # 문서의 검색 결과
    answer: Annotated[str, "answer"]            # 답변
    relevance: Annotated[str, "relevance"]      # 관련성
    llm_agent: Annotated[str, "llm_agent"]  #  질문에 해당하는 LLM 에이전트

# Retiever
vectorstore = chroma_vectorstore()
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 10}
)

# 라우터 노드 정의
# 라우터 노드는 질문 유형을 분류하고, 해당 유형에 맞는 LLM 에이전트를 결정합니다.
def router_node(state: MyState) -> MyState:
    print("---라우터 노드 실행---")
    question = state["question"]

    # LLM을 사용하여 질문 유형 분류
    router_llm = llm

    router_prompt = PromptTemplate.from_template("""
            당신은 발로란트 게임 내에서 유저의 질문 유형을 분류하는 AI입니다. 다음 질문을 분석하여 해당 질문의 유형을 결정하세요. 오직 분류 결과만 출력하세요. \n
            질문: {question}\n\n질문 유형을 다음 중 하나로 분류하세요: 'game_tutorial', 'agent_infomation', 'unclear'
            """)


    router_chain = router_prompt | router_llm | StrOutputParser()

    question_type = router_chain.invoke({"question": question})
    print(f"질문 유형: {question_type.strip()}")

    # 다음 노드를 결정하여 상태에 저장
    if "game_tutorial" in question_type.lower():
        state['llm_agent'] = "game_tutorial"
    elif "agent_infomation" in question_type.lower():
        state['llm_agent'] = "retriever"  # 에이전트 정보 질문 처리 노드
    else:
        state['llm_agent'] = "unclear_handler" # 분류 불가능한 경우 처리
    return state



# 라우터 이후의 노드들을 정의합니다
#====== 라우터 이후의 노드들 정의 ======

# 게임 튜토리얼 질문을 처리하는 노드
def game_tutorial_node(state: MyState) -> MyState:
    answer=tutorial_chain.invoke(state["question"])
    print("---게임 튜토리얼 질문 처리---")
    state['answer'] = answer
    return state

# 에이전트 정보 질문을 처리하는 노드
# Retiever Node -> context
def retrieve_node(state: MyState) -> MyState:
    print("---에이전트 정보 검색 문서 검색 노드 실행---")
    """
    주어진 질문을 기반으로 문서를 검색하여 상태의 'context'를 업데이트합니다.
    """
    question = state["question"]
    documents = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in documents])

    return {"context": context}

# LLM Answer Node -> evlauate
def answer_node(state: MyState) -> MyState:
    print("---답변 생성 노드 실행---")
    """
    주어진 Context를 기반으로 질문에 기반한 답변을 생성하여 상태의 'answer'를 업데이트합니다.
    """
    question = state["question"]
    context = state["context"]

    # Prompt Template Define
    # prompt_template = PromptTemplate.from_template([
    #     ("system", "당신은 게임 개발조직의 유용한 AI Assitant입니다. 다음 참조 문서와 질문을 바탕으로 답변을 제공하세요. 답변은 참조 문서의 정보에만 근거해야 하며, 모르는 내용은 '모르는 내용입니다'라고 답변하세요. 사용자에게 필요한 정보를 명확하고 간결하게 전달해 주세요. 답변은 2-3문장 정도로 작성하세요."),
    #     ("user", "참조 문서: \n{context}\n\n질문: {question}")
    # ])
    prompt_template = PromptTemplate.from_template(
        "당신은 게임 개발조직의 유용한 AI Assitant입니다. 다음 참조 문서와 질문을 바탕으로 답변을 제공하세요. "
        "답변은 참조 문서의 정보에만 근거해야 하며, 모르는 내용은 '모르는 내용입니다'라고 답변하세요. "
        "사용자에게 필요한 정보를 명확하고 간결하게 전달해 주세요. 답변은 2-3문장 정도로 작성하세요.\n\n"
        "참조 문서: \n{context}\n\n질문: {question}"
    )
    # RAG Chain Composition
    rag_chain = (
        RunnableMap({
            "question": RunnablePassthrough(),
            "context": RunnablePassthrough()
        })
        | prompt_template
        | llm
        | StrOutputParser()
    )
    answer = rag_chain.invoke({"question": question, "context": context})
    
    return {"answer": answer}


# 분류 불가능한 질문을 처리하는 노드
def unclear_handler_node(state: MyState) -> MyState:
    print("---분류 불가능한 질문 처리---")
    state['answer'] = "죄송합니다. 질문의 의도를 명확히 파악하기 어렵습니다. 좀 더 자세히 설명해 주시겠어요?"
    return state

# Evaluate Node(추가예정)/ (임시 간단 평가 함수)

def simple_evaluate_Node(state: MyState) -> MyState:
    print("""간단한 답변 품질 평가""")
    question = state["question"]
    answer = state["answer"]
    try:
        eval_prompt = f"""
        다음 답변을 평가해주세요:
        질문: {question}
        답변: {answer}
        관련성, 충실성, 명확성, 간결성 측면에서 평가해주세요. 
        각 항목을 고려하여 질문과 답변을 분석하고, 평가 결과를 '적절', '부적절'로 구분하여 출력해주세요.
        오직 평가 결과만 출력해주세요.
        """
        eval_result = llm.invoke(eval_prompt).content
        if '적절' in eval_result:
            eval_result = "적절"
        elif '부적절' in eval_result:
            eval_result = "부적절"
        state['relevance'] = eval_result
        print(f"답변 평가 내역: {state['relevance']}")
        return state
    except Exception as e:
        state['relevance'] = "부적절"
        print(f"답변 평가 내역: {state['relevance']}")
        return state

# StateGraph 인스턴스 생성
graph = StateGraph(MyState)

# Node Define
graph.add_node("retriever", retrieve_node)
graph.add_node("Answer", answer_node)
graph.add_node("router", router_node)  # 라우터 노드 추가
graph.add_node("game_tutorial", game_tutorial_node)  # 게임 튜토리얼 질문 처리 노드    
graph.add_node("unclear_handler", unclear_handler_node)  # 분류 불가능한 질문 처리 노드
graph.add_node("simple_evaluate", simple_evaluate_Node)  # 간단한 평가 노드

# Edge Define
graph.add_edge(START, "router")
graph.add_edge("game_tutorial", "simple_evaluate")
graph.add_edge("retriever", "Answer")
graph.add_edge("Answer", "simple_evaluate")
graph.add_edge("unclear_handler", END)

# graph.add_edge(START, "game_tutorial")  # 시작 노드에서 라우터 노드로 이동
# graph.add_edge("game_tutorial", END)  # 게임 튜토리얼 노드에서 평가 노드로 이동
graph.add_edge("simple_evaluate", END)  # 평가 노드에서 종료 노드로 이동
# Conditional Edges Define

# 조건부 엣지를 추가하여 라우터 노드에서 다음 노드를 결정합니다.
def route_decision(state: MyState) -> str:
    """라우터 노드에서 다음 노드를 결정하는 함수"""
    return state["llm_agent"]

# 라우터 노드가 실행된 후, route_decision 함수가 다음 노드를 결정합니다.
graph.add_conditional_edges(
    "router",          # 이 노드에서 출발
    route_decision,    # 이 함수가 다음 노드를 결정
    {
        "game_tutorial": "game_tutorial",
        "retriever": "retriever",
        "unclear_handler": "unclear_handler",
    }
)
# simple_evaluate 노드에서 END로 가는 엣지는 이미 위에서 정의되어 있으므로, 추가로 조건부 엣지는 필요하지 않습니다.
# 만약 평가 결과에 따라 분기하고 싶다면 아래와 같이 조건부 엣지를 추가할 수 있습니다.

def evaluate_decision(state: MyState) -> str:
    """simple_evaluate 노드에서 평가 결과에 따라 다음 노드를 결정"""
    if state.get("relevance") == "적절":
        return END
    else:
        return "router"  # 평가가 부적절한 경우 다시 라우터로 돌아갑니다.

graph.add_conditional_edges(
    "simple_evaluate",
    evaluate_decision,
    {
        END: END,
        "router": "router"  # 평가가 부적절한 경우 다시 라우터로 돌아갑니다.
    }
)

# Compile
graph_app = graph.compile()

# Response
if __name__ == "__main__":
    response = graph_app.invoke({"question": "각 캐릭터의 특징을 알려줘"})
    print(response)