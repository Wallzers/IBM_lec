import os
import math
import streamlit as st
from pydub import AudioSegment
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import google.generativeai as genai

# Initialize Gemini
genai.configure(api_key=os.getenv("AIzaSyA8324YoI8wxnihUqRzbweAwYgVpP73IX0"))
# -------------------------------
# Load Whisper ASR
# -------------------------------
@st.cache_resource
def load_whisper():
    model_id = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
    )
    return asr


# -------------------------------
# Transcribe with chunking
# -------------------------------
def transcribe_audio(audio_path, whisper_model, chunk_length_ms=30000):
    audio = AudioSegment.from_file(audio_path)
    num_chunks = math.ceil(len(audio) / chunk_length_ms)
    transcript = ""

    progress = st.progress(0)
    status = st.empty()

    for i in range(num_chunks):
        start = i * chunk_length_ms
        end = min((i + 1) * chunk_length_ms, len(audio))
        chunk = audio[start:end]
        chunk_path = f"chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        result = whisper_model(chunk_path)
        transcript += " " + result["text"]

        os.remove(chunk_path)

        progress.progress((i + 1) / num_chunks)
        status.text(f"Processed chunk {i+1}/{num_chunks}")

    progress.empty()
    status.text("✅ Transcription complete!")
    return transcript.strip()


# -------------------------------
# Summarization + Flashcards (Ollama)
def summarize_and_quiz_gemini(text):
    model = genai.GenerativeModel("gemini-2.5-flash")

    # Summarization
    summary_prompt = (
        "Summarize the following lecture into clear, well-structured study notes:\n\n" + text
    )
    summary_response = model.generate_content(summary_prompt)
    summary = summary_response.text

    # Flashcards / Quizzes
    quiz_prompt = (
        "Generate 5 flashcards (question and concise answer) based on this lecture:\n\n" + text
    )
    quiz_response = model.generate_content(quiz_prompt)
    flashcards = quiz_response.text

    return summary, flashcards

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Lecture Voice-to-Notes", layout="wide")
st.title("🎙️ Lecture Voice-to-Notes Generator")

uploaded_file = st.file_uploader(
    "Upload lecture audio (mp3, wav, m4a, mp4)",
    type=["mp3", "wav", "m4a", "mp4"],
)

if uploaded_file is not None:
    with open("temp_audio", "wb") as f:
        f.write(uploaded_file.read())

    st.info("⏳ Transcribing lecture... please wait.")
    whisper_model = load_whisper()
    transcript = transcribe_audio("temp_audio", whisper_model)

    st.subheader("📝 Transcript")
    st.write(transcript)

    st.info("⏳ Generating summary and flashcards with Mistral...")
    summary, flashcards = summarize_and_quiz_gemini(transcript)

    st.subheader("📌 Study Notes")
    st.write(summary)

    st.subheader("🃏 Flashcards / Quiz")
    st.write(flashcards)


    #python -m streamlit run app.py
