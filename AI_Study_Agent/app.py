# app.py
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document
import openai
import sqlite3
import pyttsx3
import speech_recognition as sr
from textblob import TextBlob

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# -----------------------
# OpenAI API Key
# -----------------------
openai.api_key = "YOUR_OPENAI_API_KEY"

# -----------------------
# SQLite DB for Doubts
# -----------------------
conn = sqlite3.connect("doubts.db")
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS doubts (
    id INTEGER PRIMARY KEY,
    question TEXT,
    answer TEXT
)
""")
conn.commit()

def save_doubt(question, answer):
    c.execute("INSERT INTO doubts (question, answer) VALUES (?, ?)", (question, answer))
    conn.commit()

def get_previous_doubts():
    c.execute("SELECT question, answer FROM doubts ORDER BY id DESC")
    return c.fetchall()

# -----------------------
# TTS Engine
# -----------------------
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# -----------------------
# Speech Recognition
# -----------------------
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio)
            st.success(f"You said: {query}")
            return query
        except Exception:
            st.error("Sorry, could not understand.")
            return ""

# -----------------------
# Spell Correction
# -----------------------
def correct_spelling(text):
    return str(TextBlob(text).correct())

# -----------------------
# File Extraction
# -----------------------
def extract_text(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    elif file.type == "text/plain":
        return str(file.read(), "utf-8")
    else:
        return None

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("📚 AI Study Assistant with RAG & Voice")

# Sidebar settings
st.sidebar.header("Settings")
chunk_size = st.sidebar.slider("Chunk Size (characters)", min_value=500, max_value=5000, value=1500, step=100)

# File upload
uploaded_file = st.file_uploader("Upload PDF, Word, or TXT file")
if uploaded_file:
    text = extract_text(uploaded_file)
    if text:
        st.subheader("Document Content Preview")
        st.text_area("Document Content", text[:2000] + ("..." if len(text) > 2000 else ""), height=300)

        # -----------------------
        # RAG: Split & Embed
        # -----------------------
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", "!", "?"]
        )
        docs = text_splitter.create_documents([text])
        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)

        # LangChain QA Chain
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-4", temperature=0.2),
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # -----------------------
        # Interactive Chat
        # -----------------------
        user_question = st.text_input("Ask a question or type 'explain' for full content")

        if st.button("Get Answer"):
            if user_question.strip() == "":
                st.warning("Please enter a question or 'explain'")
            else:
                corrected_question = correct_spelling(user_question)
                answer = qa.run(corrected_question)
                st.subheader("AI Answer")
                st.text_area("Answer", answer, height=300)
                speak(answer)
                save_doubt(corrected_question, answer)

        # Voice Question
        if st.button("Ask via Voice"):
            voice_input = listen()
            if voice_input:
                corrected_voice = correct_spelling(voice_input)
                answer = qa.run(corrected_voice)
                st.text_area("AI Answer", answer, height=300)
                speak(answer)
                save_doubt(corrected_voice, answer)

        # Show previous doubts
        if st.checkbox("Show Previous Questions & Answers"):
            doubts = get_previous_doubts()
            for q, a in doubts:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.write("---")
    else:
        st.error("Unsupported file type")
