import streamlit as st

st.title("Bar Exam Indexer - Test App")

uploaded_file = st.file_uploader("Upload your exam materials (PDF)", type="pdf")
if uploaded_file:
    st.write(f"Uploaded file: {uploaded_file.name}")
import fitz  # PyMuPDF

if uploaded_file:
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page_num, page in enumerate(doc, start=1):
        text += f"--- Page {page_num} ---\n"
        text += page.get_text()
    st.text_area("PDF Text Preview", text, height=400)
