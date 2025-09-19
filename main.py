# from fastapi import FastAPI, UploadFile, Form
# from typing import Optional
# from Main_rag_teaching_assistant import LESSON_FILE, TeachingAssistant, Config

# app = FastAPI()
# config = Config()
# assistant = TeachingAssistant(config)

# @app.get("/")
# def health():
#     return {"status": "Nephele 3.0 API Running"}

# @app.post("/load_document/")
# def load_document(source: str = Form(...)):
#     success = assistant.load_document(source)
#     return {"success": success}

# @app.post("/teach_lesson/")
# def teach_lesson(topic: str = Form(...)):
#     lesson_success = assistant.teach_lesson(topic)
#     lesson_text = assistant.load_lesson_from_file()
#     return {"success": lesson_success, "lesson": lesson_text}

# @app.post("/doubt/")
# # def doubt(question: str = Form(...)):
# #     is_related = assistant.is_question_related_with_llm(
# #         question,  # question
# #         LESSON_FILE,  # lesson_filename
# #         assistant.config.openai_api_key,  # api_key
# #         model=assistant.config.model_name
# #     )
# #     if is_related:
# #         context = "\n\n".join(assistant.rag_pipeline.retrieve_relevant_chunks(question))
# #         answer = assistant.llm_provider.answer_question(question, context)
# #         return {"related": True, "answer": answer}
# #     else:
# #         return {"related": False, "answer": "This topic will be covered in future lessons."}

# @app.post("/doubt/")
# def doubt(question: str = Form(...)):
#     is_related = assistant.is_question_related_with_llm(
#         question,
#         "W:/anaconda/18-5-25/project_folder/last_lesson.txt",
#         assistant.config.openai_api_key,
#         model=assistant.config.model_name
#     )
#     if is_related:
#         context_chunks = assistant.rag_pipeline.retrieve_relevant_chunks(question)
#         context = "\n\n".join(context_chunks) if context_chunks else assistant.load_lesson_from_file()
#         answer = assistant.llm_provider.answer_question(question, context)
#         assistant.audio_manager.speak_text(answer)  # <-- THIS LINE plays the answer!
#         return {"related": True, "answer": answer}
#     else:
#         msg = "This topic will be covered in future lessons."
#         assistant.audio_manager.speak_text(msg)
#         return {"related": False, "answer": msg}

# # Extra: To add PDF upload (optional, not tested in your full code)
# @app.post("/upload_pdf/")
# async def upload_pdf(file: UploadFile):
#     save_path = f"W:/anaconda/18-5-25/{file.filename}"
#     with open(save_path, "wb") as f:
#         f.write(await file.read())
#     success = assistant.load_document(save_path)
#     return {"success": success}

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import os
from Main_rag_teaching_assistant import TeachingAssistant, Config, AUDIO_FILES_DIR

app = FastAPI()
config = Config()
assistant = TeachingAssistant(config)

# 1. Health check
@app.get("/")
def health():
    return {"status": "Nephele 3.0 API Running"}
                                                                            # http://localhost:8000/docs#/ to run the swagger ui
# 2. Usual endpoints for lesson loading/teaching (for completeness)
@app.post("/load_document/")
def load_document(source: str = Form(...)):
    success = assistant.load_document(source)
    return {"success": success}

@app.post("/teach_lesson/")
async def teach_lesson(file: UploadFile = File(...)):
    # Save & transcribe topic audio
    audio_path = os.path.join(AUDIO_FILES_DIR, f"teach_{file.filename}")
    with open(audio_path, "wb") as f_out:
        f_out.write(await file.read())
    topic = assistant.speech_recognizer.transcribe_file(audio_path)
    lesson_success = await assistant.teach_lesson(topic)   # << Only call once!
    lesson_text = assistant.load_lesson_from_file()
    return {
        "topic": topic,
        "success": lesson_success,
        "lesson": lesson_text
    }

from uuid import uuid4

@app.post("/doubt/")
async def doubt(file: UploadFile = File(...)):
    audio_path = os.path.join(AUDIO_FILES_DIR, f"doubt_{file.filename}")
    with open(audio_path, "wb") as f_out:
        f_out.write(await file.read())
    question = assistant.speech_recognizer.transcribe_file(audio_path)
    is_related = assistant.is_question_related_with_llm(
        question,
        "W:/anaconda/18-5-25/project_folder/last_lesson.txt",
        assistant.config.openai_api_key,
        model=assistant.config.model_name
    )
    if is_related:
        context_chunks = assistant.rag_pipeline.retrieve_relevant_chunks(question)
        context = "\n\n".join(context_chunks) if context_chunks else assistant.load_lesson_from_file()
        answer = assistant.llm_provider.answer_question(question, context)
    else:
        answer = "This topic will be covered in future lessons."

    output_filename = f"response_{uuid4().hex}.mp3"
    audio_file = await assistant.audio_manager.text_to_speech(answer, output_filename)
    if audio_file and os.path.exists(audio_file):
        assistant.audio_manager.play_audio_blocking(audio_file)
    audio_filename = os.path.basename(audio_file) if audio_file else None
    return {
        "question": question,
        "related": is_related,
        "answer": answer,
        "audio_file": audio_filename
    }



# @app.post("/teach_lesson/")
# async def teach_lesson(file: UploadFile = File(...)):
#     # Save & transcribe topic audio
#     audio_path = os.path.join(AUDIO_FILES_DIR, f"teach_{file.filename}")
#     with open(audio_path, "wb") as f_out:
#         f_out.write(await file.read())
#     topic = assistant.speech_recognizer.transcribe_file(audio_path)
#     lesson_success = await assistant.teach_lesson(topic)
#     lesson_text = assistant.load_lesson_from_file()

#     # Generate narration mp3s (NO PLAYBACK as async—play ALL out loud here)
#     paragraphs = [p.strip() for p in lesson_text.split('\n\n') if p.strip()]
#     for i, paragraph in enumerate(paragraphs):
#         if len(paragraph) > 20:
#             # Generate mp3
#             audio_file = await assistant.audio_manager.text_to_speech(paragraph, f"lesson_part_{i}.mp3")
#             if audio_file and os.path.exists(audio_file):
#                 assistant.audio_manager.play_audio_blocking(audio_file)  # << This plays each section!
#     return {
#         "topic": topic,
#         "success": lesson_success,
#         "lesson": lesson_text
#     }



# 4. Endpoint to download/stream the mp3 file
# @app.get("/audio/{filename}")
# def get_audio(filename: str):
#     file_path = os.path.join(AUDIO_FILES_DIR, filename)
#     if not os.path.isfile(file_path):
#         return JSONResponse(status_code=404, content={"error": "File not found"})
#     return FileResponse(file_path, media_type="audio/mpeg")

# 5. Cleanup endpoint to delete all mp3/wav files in audio_files folder
@app.post("/cleanup_audio_files/")
def cleanup_audio_files():
    deleted = []
    try:
        for file in os.listdir(AUDIO_FILES_DIR):
            if file.endswith(".mp3") or file.endswith(".wav"):
                full_path = os.path.join(AUDIO_FILES_DIR, file)
                os.remove(full_path)
                deleted.append(file)
        return {"deleted_files": deleted, "status": "Cleaned up audio files!"}
    except Exception as e:
        return {"deleted_files": deleted, "error": str(e)}

