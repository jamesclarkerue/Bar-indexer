import streamlit as st

st.title("Bar Exam Indexer - Test App")

uploaded_file = st.file_uploader("Upload your exam materials (PDF)", type="pdf")
if uploaded_file:
    st.write(f"Uploaded file: {uploaded_file.name}")
