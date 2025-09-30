import streamlit as st
import PyPDF2
from gtts import gTTS
from transformers import pipeline
import tempfile

# Load summarizer once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")  # smaller model for faster load

summarizer = load_summarizer()

# Streamlit UI
st.set_page_config(page_title="PDF to Audio Summarizer", page_icon="📘")
st.title("📘 PDF to Audio Summarizer")
st.write("Upload a PDF, summarize it, and convert it to audio in your chosen language!")

# Language selection
language = st.selectbox(
    "Choose audio language:",
    options=["English", "Hindi", "Spanish", "French"],
    index=0
)

# Map for gTTS language codes
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr"
}

# Upload PDF
pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

if pdf_file:
    # Read PDF text
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    if text.strip() == "":
        st.warning("⚠️ This PDF has no readable text.")
    else:
        # Summarize PDF
        with st.spinner("Summarizing..."):
            summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        st.success("✅ Summary Completed!")
        st.subheader("Summary:")
        st.write(summary)

        # Convert summary to audio in selected language
        with st.spinner(f"Converting summary to audio ({language})..."):
            tts = gTTS(summary, lang=lang_map[language])
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_file.name)

        st.audio(temp_file.name, format="audio/mp3")
        st.download_button(
            label="Download Audio",
            data=open(temp_file.name, "rb").read(),
            file_name=f"summary_audio_{language}.mp3",
            mime="audio/mp3"
        )
