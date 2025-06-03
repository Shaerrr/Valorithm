from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
import os
from langchain_openai import OpenAIEmbeddings

# .env 파일에서 환경 변수 로드
load_dotenv()

# 파일 로드
file_path = "data/rag_data_valorant.pdf"
loader = PyMuPDFLoader(file_path)
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

# openai_embeddings 
openai_embeddings = OpenAIEmbeddings()

# Chroma Vector DB
def chroma_vectorstore():
    vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=openai_embeddings, # or hf_embeddings,
    persist_directory="./vectorstore")

    return vectorstore

# Save on Local
if __name__ == "__main__":
    vectorstore = chroma_vectorstore()
    vectorstore.persist()
    print("Vectorstore saved successfully.")