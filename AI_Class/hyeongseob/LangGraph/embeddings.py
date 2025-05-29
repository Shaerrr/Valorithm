from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
import os

# .env 파일에서 환경 변수 로드
load_dotenv()

# 파일 로드
loader = PyPDFLoader("AI_Class/hyeongseob/LangGraph/data/rag_data_valorant.pdf")
doc = loader.load()

# 파일 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["\n\n", "\n", ".", " ", ""]
    )
split_docs = text_splitter.split_documents(doc)

# Embeddings Model
model = "intfloat/multilingual-e5-large-instruct"
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=model,
    task="feature-extraction",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

# Chroma Vector DB
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=hf_embeddings,
    persist_directory="AI_Class/hyeongseob/LangGraph/vectorstore"
)

# Save on Local
vectorstore.persist()