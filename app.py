import streamlit as st
import PyPDF2
from gtts import gTTS
from transformers import pipeline
import tempfile

# Load summarizer once
@st.cache_resource
def load_summarizer():
    return pipeline("summarization")

summarizer = load_summarizer()

# Streamlit UI
st.set_page_config(page_title="PDF to Audio Summarizer", page_icon="📘")
st.title("📘 PDF to Audio Summarizer")
st.write("Upload a PDF, summarize it, and convert it to audio!")

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
        # Summarize
        if st.button("Summarize PDF"):
            with st.spinner("Summarizing..."):
                summary = summarizer(text, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
            st.success("✅ Summary Completed!")
            st.subheader("Summary:")
            st.write(summary)

            # Convert to Audio
            if st.button("Convert Summary to Audio"):
                with st.spinner("Converting to audio..."):
                    tts = gTTS(summary)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(temp_file.name)

                st.audio(temp_file.name, format="audio/mp3")
                st.download_button(
                    label="Download Audio",
                    data=open(temp_file.name, "rb").read(),
                    file_name="summary_audio.mp3",
                    mime="audio/mp3"
                )
