# ===================== Bar Exam Indexer (Ontario) =====================
# Streamlit app: upload PDF -> exam-style issues w/ pages -> edit -> cheat sheets
# =====================================================================

import os, io, json, re
from typing import List, Dict, Tuple

import streamlit as st
import pandas as pd
import fitz  # PyMuPDF

# ===================== Subject presets =====================
SUBJECT_PRESETS = { ... }   # dictionary with examples + authority hints

ISSUE_STYLE_EXAMPLES = """..."""
ISSUE_SCHEMA = """..."""

def subject_preset_block(subject: str) -> str:
    ...
def _granularity_text(level: str) -> str:
    ...
def build_chunk_prompt(...):
    ...
def build_cheat_prompt(...):
    ...

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ===================== Config =====================
st.set_page_config(page_title="Bar Exam Indexer — Ontario", layout="wide")
st.title("Bar Exam Indexer • Ontario")

# ===================== OpenAI client =====================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = None
if api_key and OpenAI:
    client = OpenAI(api_key=api_key)

if not client:
    st.error("No OpenAI API key configured. Add OPENAI_API_KEY in Streamlit Secrets.")

# ===================== Subject presets =====================
SUBJECT_PRESETS = {
    "Professional Responsibility": {
        "examples": """
- Conflict of interest — bright-line rule (current vs former client)
- Confidentiality — exceptions (imminent risk; court order)
- Competence & scope — accepting/withdrawing retainer
- Duty of candour to the tribunal — correcting false evidence
- Trust accounting — mixed funds; record-keeping
- Advertising & solicitation — permissible claims; referrals
""",
        "authorities_hint": "LSO Rules of Professional Conduct; By-Laws; discipline cases."
    },
    "Civil": {
        "examples": """
- Appeal route from SCJ to Divisional Court — Civil
- Summary judgment — no genuine issue requiring a trial (RCP r. 20.04)
- Motion to strike vs judgment on pleadings — Rule 21 vs Rule 25
- Security for costs — entitlement and factors (R. 56)
- Interlocutory injunction — RJR-MacDonald test
""",
        "authorities_hint": "Rules of Civil Procedure; Courts of Justice Act; Limitations Act, 2002; ON/CA cases."
    },
    "Criminal": {
        "examples": """
- Search incident to arrest — scope limits
- Detention vs arrest — s. 9 & 10(b) cautions
- Exclusion of evidence — Grant framework (s. 24(2))
- Bail — ladder principle; tertiary ground
- Mens rea for murder vs manslaughter — objective vs subjective
""",
        "authorities_hint": "Criminal Code; Charter; SCC/ONCA cases (Grant, St-Onge Lamoureux, etc.)."
    },
    "Family": {
        "examples": """
- Best interests test — parenting time/decision-making (CLRA/Divorce Act)
- Mobility — Gordon test; material change
- Child support — table amounts vs undue hardship
- Spousal support — entitlement; SSAG ranges
- Division of property — equalization; excluded assets
""",
        "authorities_hint": "Divorce Act; CLRA; FLA; SSAG; ONCA/SCC cases."
    },
    "Public": {
        "examples": """
- Judicial review — prematurity; adequate alternative remedy
- Standard of review — reasonableness vs correctness (Vavilov)
- Procedural fairness — duty trigger & content (Baker factors)
- Delegated legislation — vires; ultra vires analysis
""",
        "authorities_hint": "Vavilov; Baker; SPPA (Ontario); JRPA; Charter."
    },
}

ISSUE_STYLE_EXAMPLES = """
Generic acceptable 'issue' phrasing:
- Appeal route from SCJ to Divisional Court — Civil
- Limitation period — discoverability under s. 5, Limitations Act, 2002
- Rule 21 motion — striking claims with no reasonable cause of action
- Anton Piller order — elements and safeguards
"""

ISSUE_SCHEMA = """Return ONLY valid JSON with this shape:
{
  "issues": [
    {
      "issue": "short exam-style issue statement",
      "triggers": ["2-5 spotting cues"],
      "rule_or_test": "concise elements or test (Ontario)",
      "authorities": ["Rules/statutes/cases"],
      "pages": [int, ...]
    }
  ]
}"""

def subject_preset_block(subject: str) -> str:
    p = SUBJECT_PRESETS.get(subject, {})
    out = ""
    if p.get("examples"):
        out += f"\nSUBJECT-SPECIFIC EXAMPLES ({subject}):\n{p['examples']}\n"
    if p.get("authorities_hint"):
        out += f"\nWhen citing authorities, prefer: {p['authorities_hint']}\n"
    return out

def _granularity_text(level: str) -> str:
    if level.startswith("Fine"):
        return "Write concrete exam issues (route/test/standard/duty/remedy), not headings."
    if level.startswith("Medium"):
        return "Write practical sub-issues useful for exam answers."
    return "High level issues; fewer items."

def build_chunk_prompt(joined_pages: str, granularity: str, subject_hint: str, require_auth: bool, subject_preset_name: str) -> str:
    need_auth = ("Include specific Ontario/Canadian authorities." if require_auth else "Do not include citations unless essential.")
    extraction_policy = (
        "You are an assistant helping a bar exam student. Extract ALL POSSIBLE LEGAL ISSUES, no matter how small. "
        "If the text is mainly instructive or TOC, infer procedural/context issues instead of saying 'none'."
    )
    preset = subject_preset_block(subject_preset_name)
    return f"""You are building an exam index for the Ontario bar ({subject_hint}).

{_granularity_text(granularity)}
Avoid TOC-style headings. Each 'issue' must read like an exam spotter, e.g., 'Appeal route from SCJ to Divisional Court — Civil'.

EXAMPLES (generic):
{ISSUE_STYLE_EXAMPLES}
{preset}

{extraction_policy}

From the following pages, extract 12–25 issues using the JSON schema. {need_auth}

{ISSUE_SCHEMA}

Pages:
{joined_pages}
"""

# ===================== Sidebar =====================
with st.sidebar:
    st.header("Settings")
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=1)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.2, 0.05)
    chunk_size = st.number_input("Pages per chunk", min_value=2, max_value=25, value=8, step=1)
    max_issues = st.number_input("Max issues to keep", min_value=10, max_value=200, value=80, step=5)

    st.header("Extraction style")
    subject_preset = st.selectbox("Subject preset", list(SUBJECT_PRESETS.keys()), index=0)
    subject_hint = st.text_input("Subject focus (optional override)", value=subject_preset)
    granularity = st.selectbox("Granularity", ["Fine (exam spotting)", "Medium", "Coarse"], index=0)
    require_authorities = st.checkbox("Include statutes/rules/cases", value=True)

# ===================== Upload PDF =====================
uploaded = st.file_uploader("Upload PDF (<= 200 MB)", type=["pdf"])
pages_text = []
if uploaded:
    doc = fitz.open(stream=uploaded.getvalue(), filetype="pdf")
    for i, page in enumerate(doc):
        pages_text.append(f"[Page {i+1}]\n{page.get_text('text')}")
    st.success(f"Loaded {len(pages_text)} pages from {uploaded.name}")

# ===================== Generate Issues =====================
if st.button("Generate issues"):
    if not pages_text:
        st.warning("Please upload a PDF first.")
    elif not client:
        st.warning("OpenAI client not configured.")
    else:
        chunks = [pages_text[i:i+chunk_size] for i in range(0, len(pages_text), chunk_size)]
        all_rows = []
        for idx, ch in enumerate(chunks, start=1):
            joined = "\n\n".join(ch)
            prompt = build_chunk_prompt(joined, granularity, subject_hint, require_authorities, subject_preset)
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1400,
            )
            out = resp.choices[0].message.content.strip()
            try:
                data = json.loads(out)
                rows = data.get("issues", [])
            except:
                rows = []
            all_rows.extend(rows)
        st.session_state["issues_rows"] = all_rows
        st.success(f"Extracted {len(all_rows)} issues.")

# ===================== Review/Edit =====================
if "issues_rows" in st.session_state and st.session_state["issues_rows"]:
    df = pd.DataFrame(st.session_state["issues_rows"])
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)
    st.session_state["issues_rows"] = edited.to_dict("records")
    st.download_button("Download index as CSV", df.to_csv(index=False), "bar_index.csv")
# -------- Cheat sheet prompt builder (subject-aware) --------
CHEAT_GENERIC_STYLE = """
Write compact exam cheat sheets. For each issue:
- Definition (1–2 lines)
- Elements / test (numbered, concise)
- Pitfalls / exceptions (bullets)
- Tiny checklist (3–5 checks)
Keep each under ~120 words. Use Ontario framing.
"""

CHEAT_SUBJECT_TIPS = {
    "Professional Responsibility": """
- Emphasize LSO Rules of Professional Conduct; by-laws; commentary.
- Include classic pitfalls: conflicts, confidentiality exceptions, candour, trust accounting, advertising, civility, supervision.
- Use verbs like "must", "may", "avoid", "disclose", "withdraw".
""",
    "Civil": """
- Prefer Rules of Civil Procedure cites (rule numbers) and CJA/Limitations Act hooks.
- Emphasize tests (RJR-MacDonald, Hryniak), timelines, burdens, standards of review, motion types.
""",
    "Criminal": """
- Anchor to Criminal Code/Charter. Stress Grant (24(2)), detention/arrest, right to counsel, voluntariness, warrants/exceptions, bail ladder.
""",
    "Family": """
- Prefer Divorce Act/CLRA/FLA + SSAG. Emphasize best interests, mobility (Gordon), support (entitlement/quantum), equalization nuances, domestic contract validity.
""",
    "Public": """
- Emphasize Vavilov/Baker frameworks: standard of review, duty of fairness (trigger & content), judicial review remedies, vires/ultra vires.
""",
}

def build_cheat_prompt(
    selected_rows: list[dict],
    subject_name: str,
    subject_hint: str,
    include_cases: bool,
) -> str:
    # assemble the issues block with structured hints (rule/test, authorities, pages)
    issues_block = "\n".join(
        "- " + r.get("issue","").strip()
        + (f"\n  Rule/Test: {r['rule_or_test']}" if r.get("rule_or_test") else "")
        + (f"\n  Authorities: {r['authorities']}" if r.get("authorities") else "")
        + (f"\n  Pages: {r['pages']}" if r.get("pages") else "")
        for r in selected_rows if r.get("issue")
    )

    cases_hint = (
        "When helpful, include 1–2 key Ontario/Canadian cases or rule/statute cites (super short parentheticals)."
        if include_cases else
        "Do not include case citations; focus on elements/tests."
    )

    subject_tips = CHEAT_SUBJECT_TIPS.get(subject_name, "")

    return f"""Create concise Ontario bar exam cheat sheets for the following **{subject_hint}** issues.

{CHEAT_GENERIC_STYLE}
Subject tips:
{subject_tips}
{cases_hint}

Issues:
{issues_block}

Return GitHub-flavored Markdown with `###` headings for each issue name.
Keep language exam-ready and precise. Do not exceed ~120 words per issue.
"""
