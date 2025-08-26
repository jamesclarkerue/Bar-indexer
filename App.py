import os
import io
import math
import textwrap
from typing import List, Dict, Any

import streamlit as st
import fitz  # PyMuPDF

# OpenAI SDK v1 style
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --------------- Page config ---------------
st.set_page_config(page_title="Bar Exam Indexer", layout="wide")
st.title("Bar Exam Indexer • Issues and Cheat Sheets")

# --------------- Secrets and client ---------------
api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
elif os.getenv("OPENAI_API_KEY"):
    api_key = os.getenv("OPENAI_API_KEY")

client = None
if OpenAI and api_key:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Could not init OpenAI client: {e}")

if client is None:
    st.warning(
        "No OpenAI API key. Set it in Streamlit Cloud secrets as OPENAI_API_KEY "
        "or in your local environment."
    )

# --------------- Session state ---------------
for key, default in {
    "pages_text": [],         # list[str] one entry per page
    "pdf_name": "",
    "issues_rows": [],        # list of dicts: {"issue": str, "pages": str, "notes": str}
    "cheat_sheets_md": "",    # consolidated markdown of cheat sheets
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# --------------- Sidebar controls ---------------
with st.sidebar:
    st.header("Settings")
    model = st.selectbox(
        "Model",
        options=["gpt-4o", "gpt-4o-mini"],
        index=0,
        help="Use 4o-mini to save cost during iteration."
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    chunk_size = st.number_input(
        "Pages per chunk",
        min_value=2,
        max_value=25,
        value=8,
        step=1,
        help="The app will summarize each chunk before merging. Adjust based on your PDF size."
    )

    max_issues = st.number_input(
        "Max issues to keep",
        min_value=10,
        max_value=200,
        value=80,
        step=5,
        help="Applies in the merge step."
    )

# --------------- PDF upload ---------------
uploaded = st.file_uploader("Upload bar materials (PDF)", type=["pdf"])
if uploaded:
    try:
        st.session_state["pdf_name"] = uploaded.name
        pdf_bytes = uploaded.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        pages_text = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            # keep a simple page marker in the stored list
            pages_text.append(f"[Page {i}]\n{text}")

        st.session_state["pages_text"] = pages_text

        st.success(f"Loaded {len(pages_text)} pages from {uploaded.name}")
    except Exception as e:
        st.error(f"Error reading PDF: {e}")

# --------------- Preview ---------------
if st.session_state["pages_text"]:
    colA, colB = st.columns([1, 1])
    with colA:
        st.caption("Quick preview")
        preview_pages = min(3, len(st.session_state["pages_text"]))
        st.text("\n\n".join(st.session_state["pages_text"][:preview_pages]))
    with colB:
        st.metric("Total pages", len(st.session_state["pages_text"]))
        st.caption("Tip: adjust Pages per chunk in the sidebar for large PDFs")

# --------------- Helpers ---------------
def chunk_pages(pages: List[str], size: int) -> List[List[str]]:
    return [pages[i:i+size] for i in range(0, len(pages), size)]

def call_openai(messages: List[Dict[str, str]], max_tokens: int = 1200) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not initialized")
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def parse_issues_lines(text: str) -> List[Dict[str, str]]:
    """
    Expect bullets like:
    - Issue name [pages: 3,5,7]
    or
    - Issue name  pages 3 to 5
    Fallback: just an issue line
    """
    rows = []
    for line in text.splitlines():
        line = line.strip("-• ").strip()
        if not line:
            continue
        # naive page capture
        pages = ""
        lower = line.lower()
        if "[pages:" in lower:
            try:
                start = lower.index("[pages:")
                end = lower.index("]", start)
                pages = line[start+7:end].replace("pages:", "").replace("page:", "").strip(": ").strip()
                issue = (line[:start] + line[end+1:]).strip(" -:;")
            except Exception:
                issue = line
        else:
            issue = line
        rows.append({"issue": issue, "pages": pages, "notes": ""})
    return rows

def dataframe_clean(df) -> List[Dict[str, str]]:
    cleaned = []
    for _, row in df.iterrows():
        issue = str(row.get("issue", "")).strip()
        pages = str(row.get("pages", "")).strip()
        notes = str(row.get("notes", "")).strip()
        if issue:
            cleaned.append({"issue": issue, "pages": pages, "notes": notes})
    return cleaned

# --------------- Generate issues ---------------
st.subheader("Step 1 • Generate issues with page references")

c1, c2 = st.columns([1, 2])

with c1:
    do_generate = st.button("Generate issues")
with c2:
    st.caption("The app will summarize chunks, then merge and deduplicate to a single list with page anchors.")

if do_generate:
    if not st.session_state["pages_text"]:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("Reading and chunking the PDF and extracting issues"):
            chunks = chunk_pages(st.session_state["pages_text"], int(chunk_size))
            chunk_summaries = []

            for idx, ch in enumerate(chunks, start=1):
                joined = "\n\n".join(ch)
                prompt = f"""You are helping a bar candidate build an issues-based index.

From the following pages, list the legal issues that a student must be able to spot on the Ontario bar exams.
Attach the page numbers in square brackets as [pages: list].
Use short crisp bullets. Avoid duplication and overly narrow sub-issues.

Pages:
{joined}

Return 15 to 30 bullets only.
"""
                try:
                    out = call_openai(
                        [
                            {"role": "system", "content": "You are a concise legal study assistant."},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=900,
                    )
                    chunk_summaries.append(out)
                except Exception as e:
                    st.error(f"OpenAI error on chunk {idx}: {e}")
                    st.stop()

            merge_prompt = f"""You will receive multiple partial issue lists with page references from the same PDF.
Merge and deduplicate them into a single list of the most useful issues for an Ontario bar exam index.
Keep page references as a comma separated list of page numbers in [pages: ...].
Prefer broader canonical issue names. Limit to about {int(max_issues)} items.

Partial lists:
{"\n\n---\n\n".join(chunk_summaries)}
"""
            try:
                merged_text = call_openai(
                    [
                        {"role": "system", "content": "You create clean, deduplicated issue indexes with page anchors."},
                        {"role": "user", "content": merge_prompt},
                    ],
                    max_tokens=1400,
                )
            except Exception as e:
                st.error(f"OpenAI error during merge: {e}")
                st.stop()

            st.session_state["issues_rows"] = parse_issues_lines(merged_text)
            st.success("Issues generated")

# --------------- Edit issues table ---------------
st.subheader("Step 2 • Review and edit the index")
st.caption("Add rows, fix names, and adjust page references. You can also paste your own content below and click Apply paste.")

with st.expander("Optional paste input"):
    paste = st.text_area(
        "Paste issue lines. One per line. Include optional [pages: ...].",
        height=120,
        placeholder="- Negligence [pages: 12, 14]\n- Duty of care\n- Standard of care [pages: 15]"
    )
    colp1, colp2 = st.columns(2)
    with colp1:
        if st.button("Apply paste"):
            if paste.strip():
                st.session_state["issues_rows"].extend(parse_issues_lines(paste))
                st.success("Pasted lines added to the table")
    with colp2:
        if st.button("Clear table"):
            st.session_state["issues_rows"] = []

# Build an editable table
import pandas as pd
issues_df = pd.DataFrame(st.session_state["issues_rows"] or [], columns=["issue", "pages", "notes"])
edited_df = st.data_editor(
    issues_df,
    num_rows="dynamic",
    use_container_width=True,
    key="issues_editor",
)

st.session_state["issues_rows"] = dataframe_clean(edited_df)

# --------------- Export index ---------------
colx, coly, colz = st.columns(3)
with colx:
    if st.session_state["issues_rows"]:
        csv_bytes = pd.DataFrame(st.session_state["issues_rows"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download index as CSV", csv_bytes, file_name="bar_index.csv", mime="text/csv")
with coly:
    if st.session_state["issues_rows"]:
        md_lines = [f"- {r['issue']}" + (f" [pages: {r['pages']}]" if r['pages'] else "") + (f" — {r['notes']}" if r['notes'] else "") for r in st.session_state["issues_rows"]]
        md_text = "\n".join(md_lines)
        st.download_button("Download index as Markdown", md_text, file_name="bar_index.md", mime="text/markdown")
with colz:
    st.write("")

# --------------- Cheat sheets ---------------
st.subheader("Step 3 • Generate cheat sheets for issues")

cols = st.columns([1, 1, 2])
with cols[0]:
    top_k = st.number_input("Number of issues to include", 3, 50, min(len(st.session_state["issues_rows"]) or [10])[0] if st.session_state["issues_rows"] else 10)
with cols[1]:
    include_cases = st.checkbox("Ask for Canadian and Ontario cases", value=True)
with cols[2]:
    st.caption("Cheat sheets are concise quick reference notes with elements, tests, cases, pitfalls, and a short checklist.")

if st.button("Generate cheat sheets"):
    if not st.session_state["issues_rows"]:
        st.warning("Create or import an index first.")
    elif client is None:
        st.warning("OpenAI client not configured.")
    else:
        with st.spinner("Writing cheat sheets"):
            selected = st.session_state["issues_rows"][: int(top_k)]
            issues_block = "\n".join(
                f"- {r['issue']}" + (f" [pages: {r['pages']}]" if r['pages'] else "")
                for r in selected
            )
            cases_hint = "Include leading Canadian and Ontario cases where relevant with super short parentheticals." if include_cases else "Do not include case citations. Focus on elements and tests."

            cheat_prompt = f"""Create concise bar exam cheat sheets for the following issues.
Use this structure for each issue:
Issue
Definition one or two lines
Elements or test in clear numbered steps
Common pitfalls or exceptions
Very short checklist for spotting and answering on exams
{cases_hint}

Issues:
{issues_block}

Keep each issue under 120 words.
Return your result as GitHub flavored Markdown with H3 headings for each issue name.
"""
            try:
                md = call_openai(
                    [
                        {"role": "system", "content": "You produce compact bar exam cheat sheets. Be accurate and concise."},
                        {"role": "user", "content": cheat_prompt},
                    ],
                    max_tokens=1800,
                )
                st.session_state["cheat_sheets_md"] = md
                st.success("Cheat sheets created")
            except Exception as e:
                st.error(f"OpenAI error while creating cheat sheets: {e}")

# --------------- Cheat sheet editor and export ---------------
if st.session_state["cheat_sheets_md"]:
    st.markdown("### Cheat sheets (editable)")
    cheat_md = st.text_area("Markdown", st.session_state["cheat_sheets_md"], height=500, key="cheat_md_area")
    st.session_state["cheat_sheets_md"] = cheat_md

    st.download_button(
        "Download cheat sheets as Markdown",
        st.session_state["cheat_sheets_md"],
        file_name="cheat_sheets.md",
        mime="text/markdown",
    )
