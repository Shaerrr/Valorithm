# 환경변수
from dotenv import load_dotenv
load_dotenv()
import os 
import sys
# 라이브러리
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# FAISS Vector Store를 사용하여 문서 검색을 위한 RAG 시스템을 구축합니다.
# PDF 문서 로드
loader = PyMuPDFLoader("data/valorant_tutorial.pdf")

docs = loader.load()

text_spliteer = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
texts = text_spliteer.split_documents(docs)

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# LLM 모델 설정
llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-04-17', temperature=0.1)

# 프롬프트 템플릿 정의
prompt = PromptTemplate.from_template(
"""
역할 : 
- 당신은 발로란트 게임에 대해 알려주는 친절한 인게임 AI 어시스턴트입니다. 주 대상은 발로란트 게임을 처음 접하는 사람들입니다.
지침 :
- 당신은 발로란트 게임의 공식 문서와 관련된 정보를 바탕으로 답변을 작성합니다.
- 모르는 내용은 "확인할 수 없습니다."라고 답변하세요.
- 인사말 등 불필요한 내용은 제외하고, 질문에 대한 답변만 작성하세요.
- 답변을 먼저 말하고, 문서 검색 결과에 대해서는 언급하지 마세요.
- 질문에 대하여 간결하고 명확한 답변을 제공하세요.
- 분량은 2-3문장 정도로 작성하세요.
---------------------------------------------------------------
다음 질문에 답변하세요: 
{question}\n\n
---------------------------------------------------------------
아래는 관련된 정보입니다:
\n{context}\n\n
답변:
"""
)
tutorial_chain ={"context":retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
tutorial_chain.invoke("팀플레이 시 중요한 점")