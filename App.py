import streamlit as st
import fitz  # PyMuPDF
import openai

st.title("Bar Exam Indexer - AI Issue Generator")

# --- Set OpenAI API key from Streamlit secrets ---
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    st.warning("OpenAI API key not found. Add it in Streamlit Cloud Secrets.")

# --- File upload ---
uploaded_file = st.file_uploader("Upload your exam materials (PDF)", type="pdf")

pdf_text = ""

if uploaded_file:
    st.write(f"Uploaded file: {uploaded_file.name}")

    try:
        # Read PDF into memory
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Extract text with page numbers
        for page_num, page in enumerate(doc, start=1):
            pdf_text += f"--- Page {page_num} ---\n"
            pdf_text += page.get_text()

        st.text_area("PDF Text Preview", pdf_text, height=300)

    except Exception as e:
        st.error(f"Error reading PDF: {e}")

# --- AI Issue Extraction ---
st.subheader("Generated Issues")
if st.button("Generate Issues"):
    if pdf_text.strip() == "":
        st.warning("Please upload a PDF first!")
    else:
        with st.spinner("Generating issues..."):
            try:
                prompt = f"Read the following legal exam material and list all possible issues as bullet points:\n\n{pdf_text}\n\nIssues:"

                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000
                )

                issues_text = response.choices[0].message.content.strip()
                edited_issues = st.text_area("Edit Issues (if needed)", issues_text, height=300)

            except Exception as e:
                st.error(f"Error generating issues: {e}")
from pdf2image import convert_from_bytes
import pytesseract
import streamlit as st

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    images = convert_from_bytes(uploaded_file.read())
    text = ""
    for i, img in enumerate(images, start=1):
        text += f"--- Page {i} ---\n"
        text += pytesseract.image_to_string(img)
    st.text_area("PDF Text Preview", text, height=300)
