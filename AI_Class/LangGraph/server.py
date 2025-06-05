from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import tempfile
import os
from pydub import AudioSegment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from fastapi import UploadFile, File
from openai import OpenAI
from main import graph_app


#==========  STT 정의 ====================================
model = whisper.load_model("medium")

def transcribe_audio(file_path: str) -> str:
    """주어진 오디오 파일을 텍스트로 변환합니다."""
    result = model.transcribe(file_path)
    return result['text'].strip()

#========================================================

def answer_to_wav(answer: str) -> str:
    speech_file_path = "answerspeech/answer_speech.mp3"
    speech_file_path_wav = "answerspeech/answer_speech.wav"
    client = OpenAI()
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="nova",
        input=answer,
        instructions="Say clearly and say it in a calm yet majestic tone, like the 'Welcome to Summoner's Rift' line from League of Legends.",
        speed=1.0,
    ) as response:
        response.stream_to_file(speech_file_path)
    # mp3 → wav 변환 (PCM 16bit, 44.1kHz, mono)
    audio = AudioSegment.from_file(speech_file_path, format="mp3")
    audio = audio.set_frame_rate(44100).set_sample_width(2).set_channels(1)
    audio.export(speech_file_path_wav, format="wav")
    return speech_file_path_wav

# llm 및 app 객체 설정, Parser 정의의
llm =llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-preview-04-17')
app = FastAPI()

#==================================================================


# 라우터 
@app.get("/")
async def read_root():
    return {"message": "Hello, Valorant Agent_bot server!"}

# 봇 응답 엔드포인트 (텍스트트)

@app.post("/botresponse")
async def convert_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        text = transcribe_audio(tmp_path)
        print(f"Transcribed text: {text}")
        os.remove(tmp_path)
        output = graph_app.invoke({"question": text})['answer']
        print(output)
        # 음성 파일 생성
        speech_file_path = answer_to_wav(output)
        print("wav 파일 생성 완료:", speech_file_path)
        # 음성 파일을 클라이언트에 반환
        return JSONResponse(
            status_code=200,
            content={"answer": output, "status": "success", "audio_file": "answerspeech\\answer_speech.wav"})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "answer": f"서버 오류 발생: {str(e)}"}
        )


# 봇 응답 엔드포인트 (음성)
@app.get("/get_audio/{sppech_file_path}")
async def get_audio(sppech_file_path: str):
    """
    음성 파일을 반환하는 엔드포인트입니다.
    """
    if os.path.exists(sppech_file_path):
        return FileResponse(sppech_file_path, media_type="audio/wav")
    else:
        return JSONResponse(
            status_code=404,
            content={"status": "error", "message": "음성 파일을 찾을 수 없습니다."}
        )
