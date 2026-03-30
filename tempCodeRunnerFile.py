import streamlit as st
import pymupdf
import io
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

#   EXTRACTING FILE
def extract_text(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    text = ""

    if ext == "pdf":
        file_bytes = uploaded_file.read()
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")

        for page in doc:
            text += page.get_text()

    else:
        st.error("Unsupported file type")

    return text

st.title("Resume Text Extractor")

    uploaded_files = st.file_uploader(
    "Upload your resume(s)",
    type=["pdf"],
    accept_multiple_files=True)
if uploaded_files:
    for file in uploaded_files:
        st.subheader(file.name)

        raw_text = extract_text(file)
        cleaned_text = clean_text(raw_text)
        final_text = remove_stopwords(cleaned_text)

        st.text_area("Final Text (Preview)", final_text[:500], height=150)