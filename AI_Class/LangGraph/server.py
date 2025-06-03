from fastapi import FastAPI
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import tempfile
import os
from fastapi import UploadFile, File


#==========  STT 정의 ====================================
model = whisper.load_model("base")

def transcribe_audio(file_path: str) -> str:
    """주어진 오디오 파일을 텍스트로 변환합니다."""
    result = model.transcribe(file_path)
    return result['text'].strip()


# llm 및 app 객체 설정, Parser 정의의
llm =llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-04-17')
app = FastAPI()

#==================================================================


# 라우터 
@app.get("/")
async def read_root():
    return {"message": "Hello, Valorant Agent!"}

@app.post("/convert-audio")
async def convert_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        text = transcribe_audio(tmp_path)
        os.remove(tmp_path)
        prompt = PromptTemplate.from_template(
            "출력은 항상 아래에 제시해주는 출력의 JSON형태로 출력해주세요.\n"
            "아래에 제시되는 명령어로부터 의미를 파악하여 아래의 출력의 형태에 맞게 채워주세요 \n"
            "도로롱 또는 도로롱과 비슷한 단어는 Lamball로 처리해주세요 \n"
            "'모두','전부'와 같은 단어가 포함되어 있다면 actor=Everyone으로 고정해주세요\n"
            "'빨리', '어서'와 같이 긴급함을 요하는 단어가 포함되면 forced=True로 해주세요. 없으면 False로 처리해주세요. \n"
            "target은 ['Stone','Tree','Ore']에서 골라주세요."
            "명령어: {command}\n"
            "출력:{format}"
            )
        parser = PydanticOutputParser(pydantic_object = JSONparser)
        prompt = prompt.partial(format=parser.get_format_instructions())
        chain = prompt | llm | parser
        output=chain.invoke(text)
        print(output)
        return output
    except:
        return {'error':"ERROR! PLEASE PRONOUNCE RIGHTLY"}