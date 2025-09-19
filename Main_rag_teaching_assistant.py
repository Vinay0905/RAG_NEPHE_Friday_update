#!/usr/bin/env python3
"""
Basic_RAG_NEphele_3.0_Enhanced_SavedLesson.py
Voice-Interactive AI Teaching Assistant with Enhanced Voice Timing and Lesson Relevance Check

Features:
- Saves taught lesson to .txt
- Answers questions/doubts only if related to saved lesson
- Tells user if question will be covered in future if not related
"""
import os
import sys
import json
import time
import asyncio
import tempfile
import threading
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import openai
import requests
from bs4 import BeautifulSoup
import PyPDF2
import speech_recognition as sr
import pygame
import edge_tts
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_chroma import Chroma
try:
    import google.generativeai as genai
except ImportError:
    genai = None
load_dotenv()
@dataclass
class Config:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    llm_provider: str = "openai"
    model_name: str = "gpt-3.5-turbo"
    temperature: float = 0.7
    max_tokens: int = 1500
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_docs: int = 5
    tts_voice: str = "en-US-AriaNeural"
    tts_rate: str = "+0%"
    tts_volume: str = "+0%"
    whisper_model: str = "whisper-large-v3"
    question_timeout: int = 30
    doubt_timeout: int = 30
    phrase_timeout: int = 10

LESSON_FILE = "W:/anaconda/18-5-25/project_folder/last_lesson.txt"
AUDIO_FILES_DIR = "W:/anaconda/18-5-25/project_folder/audio_files"

class DocumentProcessor:
    @staticmethod
    def load_from_url(url: str) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            content_selectors = ['article', 'main', '.content', '#content', '.post', '.entry']
            text = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    text = ' '.join([elem.get_text() for elem in elements])
                    break
            if not text:
                text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            raise Exception(f"Error loading URL {url}: {str(e)}")

    @staticmethod
    def load_from_pdf(file_path: str) -> str:
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            raise Exception(f"Error loading PDF {file_path}: {str(e)}")
class RAGPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len
        )
        self.vectorstore = None
        self.documents = []

    def process_document(self, text: str, source: str = "document") -> None:
        try:
            chunks = self.text_splitter.split_text(text)
            self.documents = [
                Document(page_content=chunk, metadata={"source": source, "chunk_id": i})
                for i, chunk in enumerate(chunks)
            ]
            self.vectorstore = Chroma.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                persist_directory=None
            )
            print(f"✅ Processed document into {len(self.documents)} chunks")
        except Exception as e:
            raise Exception(f"Error processing document: {str(e)}")

    def retrieve_relevant_chunks(self, query: str) -> List[str]:
        if not self.vectorstore:
            return []
        try:
            docs = self.vectorstore.similarity_search(
                query,
                k=self.config.top_k_docs
            )
            chunks = [doc.page_content for doc in docs]
            return chunks
        except Exception as e:
            print(f"⚠️ Error retrieving chunks: {str(e)}")
            return []
class LLMProvider:
    def __init__(self, config: Config):
        self.config = config
        self.setup_provider()

    def setup_provider(self):
        if self.config.llm_provider == "openai":
            openai.api_key = self.config.openai_api_key
            self.client = openai.OpenAI(api_key=self.config.openai_api_key)
        elif self.config.llm_provider == "gemini" and genai:
            genai.configure(api_key=self.config.gemini_api_key)
            self.client = genai.GenerativeModel('gemini-pro')

    def generate_lesson(self, topic: str, context: str) -> str:
        prompt = f"""
        You are an expert teacher creating an engaging lesson. Based on the provided context, create a comprehensive lesson about "{topic}".

        Context from document:
        {context}

        Please create a lesson that:
        1. Introduces the topic clearly
        2. Explains key concepts step by step
        3. Uses examples and analogies
        4. Is engaging and educational
        5. Is suitable for voice narration (clear sentences, good flow)

        Structure the lesson in clear sections that can be narrated sequentially.
        """
        return self._call_llm(prompt)

    def answer_question(self, question: str, context: str) -> str:
        prompt = f"""
        You are a knowledgeable teaching assistant. Answer the following question based on the provided context.

        Context:
        {context}

        Question: {question}

        Please provide a clear, accurate answer based on the context. If the context doesn't contain enough information, say so clearly.
        """
        return self._call_llm(prompt)

    def _call_llm(self, prompt: str) -> str:
        try:
            if self.config.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.choices[0].message.content
            elif self.config.llm_provider == "gemini" and genai:
                response = self.client.generate_content(prompt)
                return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"
# This class likely manages audio-related tasks and functionalities.


class AudioManager:
    _audio_files = []  # Class-level tracker for all generated audio files

    def __init__(self, config):
        self.config = config
        self.audio_dir = AUDIO_FILES_DIR
        os.makedirs(self.audio_dir, exist_ok=True)
        # Avoid initializing pygame in API—do it only if using for CLI/local playback
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        except Exception:
            pass
        self.is_playing = False
        self.playback_lock = threading.Lock()

    async def text_to_speech(self, text: str, filename: str) -> str:
        """
        Asynchronously generate TTS mp3 file and return its path.
        """
        try:
            communicate = edge_tts.Communicate(
                text,
                self.config.tts_voice,
                rate=self.config.tts_rate,
                volume=self.config.tts_volume
            )
            if not filename.endswith(".mp3"):
                filename += ".mp3"
            filepath = os.path.join(self.audio_dir, filename)
            await communicate.save(filepath)
            AudioManager._audio_files.append(filepath)
            return filepath
        except Exception as e:
            print(f"⚠️ TTS Error: {str(e)}")
            return None

    def play_audio_blocking(self, filepath: str) -> bool:
        # Playback is only for CLI/local use, not via API!
        with self.playback_lock:
            try:
                self.is_playing = True
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    pygame.time.wait(100)
                self.is_playing = False
                return True
            except Exception as e:
                print(f"⚠️ Audio playback error: {str(e)}")
                self.is_playing = False
                return False

    async def narrate_lesson(self, lesson_text: str) -> bool:
        """
        Asynchronously generates and saves mp3 files for each section but doesn't play them.
        (API should not try to play audio.)
        """
        try:
            paragraphs = [p.strip() for p in lesson_text.split('\n\n') if p.strip()]
            print(f"🎙️ Starting narration of {len(paragraphs)} sections...")
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 20:
                    print(f"📖 Narrating section {i+1}/{len(paragraphs)}")
                    await self.text_to_speech(paragraph, f"lesson_part_{i}.mp3")
                    # No playback in API
                    time.sleep(0.5)
            print("✅ Lesson narration completed!")
            return True
        except Exception as e:
            print(f"❌ Error during narration: {str(e)}")
            return False

    async def speak_text(self, text: str) -> str:
        """
        Asynchronously generate a TTS mp3 for the answer and return the file path.
        """
        if not text or text.strip() == "":
            print("⚠️ No text to speak")
            return None
        filename = f"response_{uuid.uuid4().hex}.mp3"
        audio_file = await self.text_to_speech(text, filename)
        return audio_file  # Don't play, just return file path

    def cleanup(self):
        try:
            pygame.mixer.quit()
        except Exception:
            pass
        AudioManager.cleanup_all()

    @classmethod
    def cleanup_all(cls):
        """Delete all generated audio files (does not delete the folder)."""
        for fname in list(cls._audio_files):
            try:
                if os.path.exists(fname):
                    os.remove(fname)
                    print(f"🧹 Cleaned up: {fname}")
            except Exception as e:
                print(f"Error cleaning up {fname}: {e}")
        cls._audio_files.clear()

        audio_dir = AUDIO_FILES_DIR
        if os.path.exists(audio_dir):
            for file in os.listdir(audio_dir):
                try:
                    full_path = os.path.join(audio_dir, file)
                    if os.path.exists(full_path) and (file.endswith(".mp3") or file.endswith(".wav")):
                        os.remove(full_path)
                        print(f"🧹 Cleaned up audiodir file: {file}")
                except Exception as e:
                    print(f"Error cleaning up audio_files/{file}: {e}")
        # FOLDER IS NEVER DELETED!

class SpeechRecognizer:
    def __init__(self, config: Config):
        self.config = config
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        if not config.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        from groq import Groq
        self.groq_client = Groq(api_key=config.groq_api_key)
        with self.microphone as source:
            print("🎤 Calibrating microphone for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
    def transcribe_file(self, audio_path: str) -> str:
        with sr.AudioFile(audio_path) as source:
            audio = self.recognizer.record(source)
        audio_data = audio.get_wav_data()
        with open(audio_path, "rb") as audio_file:
            transcription = self.groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language="en"
            )
        return transcription.text.strip()

    def listen_for_speech(self, prompt: str = "Listening...", timeout: int = None, phrase_timeout: int = None) -> Optional[str]:
        try:
            if timeout is None:
                timeout = self.config.question_timeout
            if phrase_timeout is None:
                phrase_timeout = self.config.phrase_timeout
            print(f"🎤 {prompt} (Timeout: {timeout}s)")
            with self.microphone as source:
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_timeout
                )
            print("🔄 Processing speech with Groq Whisper...")
            audio_data = audio.get_wav_data()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            try:
                with open(temp_audio_path, "rb") as audio_file:
                    transcription = self.groq_client.audio.transcriptions.create(
                        file=audio_file,
                        model="whisper-large-v3",
                        language="en"
                    )
                text = transcription.text.strip()
                print(f"📝 Recognized: '{text}'")
                return text
            finally:
                try:
                    os.unlink(temp_audio_path)
                except:
                    pass
        except sr.WaitTimeoutError:
            print(f"⏱️ No speech detected within {timeout} seconds")
            return None
        except sr.UnknownValueError:
            print("❓ Could not understand speech")
            return None
        except Exception as e:
            print(f"⚠️ Speech recognition error: {str(e)}")
            return None

    def listen_for_question(self, prompt: str = "Ask your question...") -> Optional[str]:
        return self.listen_for_speech(
            prompt=f"{prompt} (Speak fully, pause 2 seconds when done)",
            timeout=self.config.question_timeout,
            phrase_timeout=None
        )

    def listen_for_doubt(self, prompt: str = "Share your doubts...") -> Optional[str]:
        return self.listen_for_speech(
            prompt=f"{prompt} (Speak fully, pause 2 seconds when done)",
            timeout=self.config.doubt_timeout,
            phrase_timeout=None
        )
class TeachingAssistant:
    def __init__(self, config: Config):
        self.config = config
        self.rag_pipeline = RAGPipeline(config)
        self.llm_provider = LLMProvider(config)
        self.audio_manager = AudioManager(config)
        self.speech_recognizer = SpeechRecognizer(config)
        self.document_loaded = False

    # --- LESSON FILE MANAGEMENT ---
    def save_lesson_to_file(self, lesson_text: str):
        with open(LESSON_FILE, "w", encoding="utf-8") as f:
            f.write(lesson_text)

    def load_lesson_from_file(self) -> str:
        try:
            with open(LESSON_FILE, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    # --- RELEVANCE CHECK ---
    # def is_question_related(self, question: str) -> bool:
    #     lesson_text = self.load_lesson_from_file()
    #     if not lesson_text or not question:
    #         return False
    #     lesson_words = set(lesson_text.lower().split())
    #     question_words = set(question.lower().split())
    #     common = lesson_words.intersection(question_words)
    #     return len(common) >= 2 # Adjust threshold if needed

    def load_sections(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        # Adjust the split logic if your files use different section headers!
        sections = [s.strip() for s in content.split('Section') if s.strip()]
        return {f"Section {i+1}": s for i, s in enumerate(sections)}

    def find_relevant_section(self,sections, question):
        vectorizer = TfidfVectorizer()
        texts = list(sections.values())
        X = vectorizer.fit_transform(texts + [question])
        sims = cosine_similarity(X[-1], X[:-1])[0]
        idx = sims.argmax()
        return list(sections.keys())[idx], texts[idx], sims[idx]

    def is_question_related_with_llm(self,question, lesson_filename, api_key, model="gpt-3.5-turbo"):
        # Load and find best matching section
        sections = self.load_sections(lesson_filename)
        section_id, section_text, similarity = self.find_relevant_section(sections, question)

        # LLM check
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        prompt = (
            f"Is the following student doubt covered or related to this lesson section?\n"
            f"Section: \"{section_text}\"\n"
            f"Doubt: \"{question}\"\n"
            "Reply 'yes' if covered, 'no' if not."
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        answer = response.choices[0].message.content.strip().lower()
        print(f"DEBUG: Best match: {section_id} (similarity={similarity:.3f}); LLM says: {answer}")
        return "yes" in answer




    def load_document(self, source: str) -> bool:
        try:
            print(f"📄 Loading document from: {source}")
            if source.startswith('http'):
                text = DocumentProcessor.load_from_url(source)
                source_name = source
            elif source.endswith('.pdf'):
                text = DocumentProcessor.load_from_pdf(source)
                source_name = os.path.basename(source)
            else:
                raise ValueError("Source must be a URL or PDF file path")
            self.rag_pipeline.process_document(text, source_name)
            self.document_loaded = True
            print(f"✅ Document loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading document: {str(e)}")
            return False

    def get_topic_input(self) -> Optional[str]:
        print("\n" + "="*50)
        print("📚 TOPIC SELECTION")
        print("="*50)
        print("💡 For voice input: End your topic with 'thank you' to finish recording")
        choice = input("Choose input method - (v)oice, (t)ext, or (q)uit: ").lower()
        if choice == 'q':
            return None
        elif choice == 'v':
            topic = self.speech_recognizer.listen_for_speech("Say the topic you'd like to learn about...")
            return topic
        elif choice == 't':
            topic = input("Enter the topic: ").strip()
            return topic if topic else None
        else:
            print("Invalid choice")
            return self.get_topic_input()

    async def teach_lesson(self, topic: str) -> bool:
        try:
            print(f"CALLED teach_lesson ONCE for topic: {topic}")

            print(f"\n🔍 Searching for relevant content about: {topic}")
            context_chunks = self.rag_pipeline.retrieve_relevant_chunks(topic)
            if not context_chunks:
                print("⚠️ No relevant content found in the document")
                return False
            context = "\n\n".join(context_chunks)
            print(f"📖 Found {len(context_chunks)} relevant sections")
            print("🤖 Generating lesson content...")
            lesson_content = self.llm_provider.generate_lesson(topic, context)
            if not lesson_content:
                print("❌ Failed to generate lesson content")
                return False
            # --- SAVE LESSON TO TEXT FILE! ---
            self.save_lesson_to_file(lesson_content)
            print(f"\n📋 Generated lesson ({len(lesson_content)} characters)")
            print("🎙️ Starting narration...")
            paragraphs = [p.strip() for p in lesson_content.split('\n\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > 20:
                    audio_file = await self.audio_manager.text_to_speech(paragraph, f"lesson_part_{i}_{uuid.uuid4().hex}.mp3")
                    if audio_file and os.path.exists(audio_file):
                        self.audio_manager.play_audio_blocking(audio_file)
            print("\n✅ Lesson completed!")
            return True
        except Exception as e:
            print(f"❌ Error teaching lesson: {str(e)}")
            return False


    # def interactive_qa(self):
    #     print("\n" + "="*50)
    #     print("❓ INTERACTIVE Q&A SESSION")
    #     print("="*50)
    #     print("Ask questions about the lesson! Say 'done' or 'exit' to finish.")
    #     print(f"⏱️ You have {self.config.question_timeout} seconds to ask each question.")
    #     while True:
    #         try:
    #             choice = input("\nAsk - (v)oice, (t)ext, or (d)one: ").lower()
    #             if choice in ['d', 'done', 'exit', 'q']:
    #                 break
    #             question = None
    #             if choice == 'v':
    #                 question = self.speech_recognizer.listen_for_question("Ask your question... (You have 30 seconds)")
    #             elif choice == 't':
    #                 question = input("Your question: ").strip()
    #             if not question:
    #                 print("No question received, continuing...")
    #                 continue
    #             if question.lower() in ['done', 'exit', 'finish', 'stop']:
    #                 break
    #             print(f"🤔 Processing question: {question}")
    #             # ---- CHECK IF QUESTION IS RELATED TO CURRENT LESSON ----
    #             if self.is_question_related(question):
    #                 context_chunks = self.rag_pipeline.retrieve_relevant_chunks(question)
    #                 context = "\n\n".join(context_chunks) if context_chunks else ""
    #                 answer = self.llm_provider.answer_question(question, context)
    #                 if answer:
    #                     print(f"💡 Answer: {answer}")
    #                     self.audio_manager.speak_text(answer)
    #                 else:
    #                     print("❌ Could not generate an answer")
    #             else:
    #                 not_relevant_msg = "This topic will be covered in future lessons."
    #                 print(not_relevant_msg)
    #                 self.audio_manager.speak_text(not_relevant_msg)
    #         except KeyboardInterrupt:
    #             break
    #         except Exception as e:
    #             print(f"⚠️ Error in Q&A: {str(e)}")
    #     print("\n✅ Q&A session ended")

    # def final_doubt_clearing_session(self):
    #     print("\n" + "=" * 60)
    #     print("🤔 FINAL DOUBT CLEARING SESSION")
    #     print("=" * 60)
    #     doubt_announcement = "If you have any doubts or questions about today's lesson, please share them now."
    #     print(f"🎙️ {doubt_announcement}")
    #     self.audio_manager.speak_text(doubt_announcement)

    #     # No explicit file deletion here—AudioManager handles cleanup globally

    #     for attempt in range(2):
    #         print(f"\n🎤 Doubt Window {attempt + 1}/2 (30 seconds each)")
    #         doubt_question = self.speech_recognizer.listen_for_doubt(
    #             f"Share your doubts now... (30 seconds - Attempt {attempt + 1}/2)"
    #         )
    #         if doubt_question:
    #             print(f"🤔 Doubt received: {doubt_question}")
    #             # Relevance check using lesson file
    #             if self.is_question_related(doubt_question, LESSON_FILE):
    #                 context_chunks = self.rag_pipeline.retrieve_relevant_chunks(doubt_question)
    #                 context = "\n\n".join(context_chunks) if context_chunks else ""
    #                 answer = self.llm_provider.answer_question(doubt_question, context)
    #                 if answer:
    #                     print(f"💡 Clarification: {answer}")
    #                     self.audio_manager.speak_text(answer)
    #                     follow_up_text = "Do you have any follow-up questions about this clarification?"
    #                     print(f"🎙️ {follow_up_text}")
    #                     self.audio_manager.speak_text(follow_up_text)
    #                     follow_up = self.speech_recognizer.listen_for_doubt("Any follow-up questions? (End with 'thank you')")
    #                     if follow_up and follow_up.lower() not in ['no', 'none', 'nothing', 'nope']:
    #                         if self.is_question_related(follow_up, LESSON_FILE):
    #                             follow_up_chunks = self.rag_pipeline.retrieve_relevant_chunks(follow_up)
    #                             follow_up_context = "\n\n".join(follow_up_chunks) if follow_up_chunks else ""
    #                             follow_up_answer = self.llm_provider.answer_question(follow_up, follow_up_context)
    #                             if follow_up_answer:
    #                                 print(f"💡 Follow-up Answer: {follow_up_answer}")
    #                                 self.audio_manager.speak_text(follow_up_answer)
    #                         else:
    #                             not_relevant_msg = "This follow-up will be covered in future lessons."
    #                             print(not_relevant_msg)
    #                             self.audio_manager.speak_text(not_relevant_msg)
    #             else:
    #                 not_relevant_msg = "This topic will be covered in future lessons."
    #                 print(not_relevant_msg)
    #                 self.audio_manager.speak_text(not_relevant_msg)
    #         else:
    #             if attempt == 0:
    #                 print("⏱️ No doubts in first window, trying second window...")
    #                 second_chance_text = "Last chance to share any doubts or questions."
    #                 print(f"🎙️ {second_chance_text}")
    #                 self.audio_manager.speak_text(second_chance_text)
    #             else:
    #                 print("⏱️ No doubts received in second window either.")
    #     final_message = "Great! It seems all concepts are clear. Let's move on to the next topic."
    #     print(f"🎙️ {final_message}")
    #     self.audio_manager.speak_text(final_message)
    #     print("\n✅ Doubt clearing session completed!")
    def final_doubt_clearing_session(self):
        print("\n" + "=" * 60)
        print("🤔 FINAL DOUBT CLEARING SESSION")
        print("=" * 60)
        doubt_announcement = "If you have any doubts or questions about today's lesson, please share them now."
        print(f"🎙️ {doubt_announcement}")
        self.audio_manager.speak_text(doubt_announcement)

        for attempt in range(2):
            print(f"\n🎤 Doubt Window {attempt + 1}/2 (30 seconds each)")

            # --- CHOOSE INPUT MODE ---
            while True:
                mode = input("Choose input method - (v)oice, (t)ext, or (q)uit: ").strip().lower()
                if mode in ['v', 't', 'q']:
                    break
                print("Invalid input. Please enter 'v', 't', or 'q'.")

            if mode == 'q':
                print("Doubt session ended by user.")
                self.audio_manager.speak_text("Doubt session ended. Let's move to the next topic.")
                return

            # --- GET DOUBT ---
            if mode == 'v':
                doubt_question = self.speech_recognizer.listen_for_doubt(
                    f"Share your doubts now... (30 seconds - Attempt {attempt + 1}/2)"
                )
            else:  # Text mode
                doubt_question = input("Enter your doubt or question: ").strip()

            if doubt_question:
                print(f"🤔 Doubt received: {doubt_question}")

                # --- CHECK RELEVANCE WITH LLM ---
                is_relevant = self.is_question_related_with_llm(
                    doubt_question,
                    LESSON_FILE,
                    self.config.openai_api_key,
                    model=self.config.model_name
                )

                if is_relevant:
                    # --- Generate 60 word LLM summary explanation ---
                    context_sections = self.rag_pipeline.retrieve_relevant_chunks(doubt_question)
                    context = "\n\n".join(context_sections) if context_sections else ""
                    prompt = (
                        f"Provide a brief clarification of the following doubt using the context below. "
                        f"Limit your answer to about 60 words.\n\n"
                        f"Context:\n{context}\n\n"
                        f"Doubt: {doubt_question}"
                    )
                    explanation = self.llm_provider._call_llm(prompt)
                    print(f"💡 Clarification: {explanation.strip()}")
                    self.audio_manager.speak_text(explanation)

                else:
                    not_relevant_msg = "This topic will be covered in future lessons."
                    print(not_relevant_msg)
                    self.audio_manager.speak_text(not_relevant_msg)

            else:
                print("No doubt received.")

            # 2nd window prompt, unless last attempt
            if attempt == 0:
                print("⏱️ No doubts in first window, trying second window...")
                second_chance_text = "Last chance to share any doubts or questions."
                print(f"🎙️ {second_chance_text}")
                self.audio_manager.speak_text(second_chance_text)
            else:
                print("⏱️ No doubts received in second window either.")

        final_message = "Great! It seems all concepts are clear. Let's move on to the next topic."
        print(f"🎙️ {final_message}")
        self.audio_manager.speak_text(final_message)
        print("\n✅ Doubt clearing session completed!")


    def run(self):
        print("🚀 Basic RAG NEphele 3.0 - Enhanced Voice-Interactive Teaching Assistant")
        print("="*80)
        try:
            if not self.document_loaded:
                source = input("Enter document source (URL or PDF path): ").strip()
                if not source:
                    print("❌ No document source provided")
                    return
                if not self.load_document(source):
                    print("❌ Failed to load document")
                    return
            while True:
                topic = self.get_topic_input()
                if topic is None:
                    print("👋 Goodbye!")
                    break
                print(f"\n🎯 Teaching topic: {topic}")
                lesson_success = self.teach_lesson(topic)
                if lesson_success:
                    # self.interactive_qa()
                    self.final_doubt_clearing_session()
                print("\n" + "="*50)
                print("🔄 Ready for next topic!")
                print("="*50)
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted by user")
        except Exception as e:
            print(f"\n❌ Application error: {str(e)}")
        finally:
            self.audio_manager.cleanup()

# def main():
#     config = Config()
#     if not config.openai_api_key and config.llm_provider == "openai":
#         print("❌ OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
#         return
#     if not config.groq_api_key:
#         print("❌ Groq API key not found. Please set GROQ_API_KEY environment variable.")
#         return
#     try:
#         assistant = TeachingAssistant(config)
#         assistant.run()
#     except Exception as e:
#         print(f"❌ Fatal error: {str(e)}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()
