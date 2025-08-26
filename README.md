# 📚 Bar Exam Indexer — Ontario Bar Prep

A Streamlit web app to help Ontario bar exam candidates quickly **index their exam materials** and build **concise cheat sheets**.  
Upload your PDF study materials, and the app will:

- 🔎 Extract legal issues (with page references)  
- ✏️ Let you edit, add notes, and export a clean **issue-based index**  
- 📑 Generate compact **cheat sheets** (elements, tests, pitfalls, cases)  
- ⬇️ Download as CSV or Markdown for study  

---

## 🚀 Live App
👉 [Click here to use the app](https://your-username-your-repo.streamlit.app)  
*(link will work once you deploy on Streamlit Cloud)*

---

## ✨ Features
- Upload PDF bar materials (up to 200 MB)  
- AI-generated **issue index with page anchors**  
- Interactive editor: add, remove, or update issues + notes  
- Cheat sheet generator: elements, exceptions, checklists, and cases  
- Export index as CSV or Markdown, export cheat sheets as Markdown  
- Sidebar controls for chunk size, model selection, and max issues  

---

## 🛠 How it Works
1. Upload your **Ontario Bar exam PDF** (Civil, Criminal, or Professional Responsibility).  
2. The app chunks the text and sends it to OpenAI’s GPT models.  
3. GPT extracts the **issues** and merges them into a clean, deduplicated index.  
4. Edit issues directly in the app (table view).  
5. Generate **cheat sheets** for quick revision.  

---

## ⚙️ Requirements
- [Streamlit](https://streamlit.io/)  
- [PyMuPDF](https://pymupdf.readthedocs.io/)  
- [OpenAI Python SDK](https://pypi.org/project/openai/)  
- Pandas  

Install locally with:

```bash
pip install -r requirements.txt
