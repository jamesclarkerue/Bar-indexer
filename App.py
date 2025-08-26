import streamlit as st
import fitz  # PyMuPDF
import openai

# --- Streamlit UI ---
st.title("Bar Exam Indexer - AI Issue Generator")

# OpenAI API key (set as environment variable in Streamlit Cloud for safety)
openai.api_key = st.secrets["OPENAI_API_KEY"]  # OR set locally: openai.api_key = "YOUR_KEY"

uploaded_file = st.file_uploader("Upload your exam materials (PDF)", type="pdf")

if uploaded_file:
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Read PDF into memory
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # Extract text with page numbers
    pdf_text = ""
    for page_num, page in enumerate(doc, start=1):
        pdf_text += f"--- Page {page_num} ---\n"
        pdf_text += page.get_text()

    # Show preview
    st.text_area("PDF Text Preview", pdf_text, height=300)

    # --- AI Issue Extraction ---
    st.subheader("Generated Issues")
    if st.button("Generate Issues"):
        prompt = f"Read the following legal exam material and list all possible issues as bullet points:\n\n{pdf_text}\n\nIssues:"
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        issues_text = response.choices[0].message.content.strip()
        # Display editable list
        edited_issues = st.text_area("Edit Issues (if needed)", issues_text, height=300)
