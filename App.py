# ===================== Bar Exam Indexer (Ontario) =====================
# Streamlit app: upload PDF -> exam-style issues w/ pages -> edit -> cheat sheets
# =====================================================================

import os, io, json, re, hashlib, datetime as dt
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
# -------- Subject presets: examples + authority hints --------
SUBJECT_PRESETS = {
    "Professional Responsibility": {
        "examples": """
- Conflict of interest — bright-line rule (current vs former client)
- Confidentiality — exceptions (imminent risk; court order)
- Competence & scope — accepting/withdrawing retainer; mandatory steps
- Duty of candour to the tribunal — correcting false evidence
- Undue influence & vulnerable client — capacity warning signs
- Trust accounting — mixed funds; record-keeping; spot audits
- Advertising & solicitation — permissible claims; referrals/fees
- Civility & communications — dealing with represented persons
- Law firm supervision — delegation; supervising non-lawyers
""",
        "authorities_hint": "LSO Rules of Professional Conduct; By-Laws; key discipline decisions."
    },
    "Civil": {
        "examples": """
- Appeal route from SCJ to Divisional Court — Civil
- Summary judgment — no genuine issue requiring a trial (RCP r. 20.04)
- Motion to strike vs judgment on pleadings — Rule 21 vs Rule 25
- Security for costs — entitlement and factors (R. 56)
- Interlocutory injunction — RJR-MacDonald test
- Anton Piller order — elements and safeguards
- Mareva injunction — freezing test and disclosure duties
- Limitation period — discoverability under s. 5, Limitations Act, 2002
- Standard of review on appeal — palpable & overriding error vs correctness
""",
        "authorities_hint": "Rules of Civil Procedure; Courts of Justice Act; Limitations Act, 2002; leading ON/CA cases (e.g., Hryniak)."
    },
    "Criminal": {
        "examples": """
- Search incident to arrest — scope limits
- Detention vs arrest — s. 9 & 10(b) cautions; right to counsel
- Exclusion of evidence — Grant framework (s. 24(2))
- Warrantless search — consent or exigent circumstances
- Admissibility of statements — voluntariness; Charter breaches
- Bail — ladder principle; tertiary ground
- Mens rea for murder vs manslaughter — objective vs subjective
- Parties to an offence — aiding/abetting; common intention
- Appeal route — summary conviction vs indictable
""",
        "authorities_hint": "Criminal Code; Charter; leading SCC/ONCA cases (Grant, St-Onge Lamoureux, etc.)."
    },
    "Family": {
        "examples": """
- Best interests test — parenting time/decision-making (CLRA/Divorce Act)
- Mobility — Gordon test; material change
- Child support — table amounts vs undue hardship
- Spousal support — entitlement (compensatory/non-compensatory); SSAG ranges
- Division of property — equalization; excluded assets; valuation date
- Domestic contracts — validity; disclosure; set-aside
- Family violence — protective orders; supervised access
- Jurisdiction/venue — interprovincial issues; forum
""",
        "authorities_hint": "Divorce Act; CLRA; FLA; SSAG; key ONCA/SCC cases."
    },
    "Public": {
        "examples": """
- Judicial review — prematurity; adequate alternative remedy
- Standard of review — reasonableness vs correctness (Vavilov)
- Procedural fairness — duty trigger & content (Baker factors)
- Bias & apprehension — reasonable apprehension test
- Delegated legislation — vires; ultra vires analysis
- Charter applicability to public bodies — s. 32
- Remedies on JR — certiorari, mandamus, prohibition
""",
        "authorities_hint": "Vavilov; Dunsmuir (context); Baker; SPPA (Ontario); JRPA; Charter."
    },
}

def subject_preset_block(subject: str) -> str:
    p = SUBJECT_PRESETS.get(subject, {})
    ex = p.get("examples", "").strip()
    hint = p.get("authorities_hint", "")
    out = ""
    if ex:
        out += f"\nSUBJECT-SPECIFIC EXAMPLES ({subject}):\n{ex}\n"
    if hint:
        out += f"\nWhen citing authorities, prefer: {hint}\n"
    return out


# ---------- OpenAI (v1 SDK) ----------
try:
    from openai import OpenAI
    OPENAI_SDK_OK = True
except Exception:
    OpenAI = None
    OPENAI_SDK_OK = False

# ---------- PDF libs ----------
import fitz  # PyMuPDF

# ===================== UI Config =====================
st.set_page_config(page_title="Bar Exam Indexer — Ontario", layout="wide")
st.title("Bar Exam Indexer • Ontario")

# ===================== Secrets / Client =====================
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
client = None
if OPENAI_SDK_OK and api_key:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Could not init OpenAI client: {e}")

# API key banner
with st.container():
    if not client:
        st.error("No OpenAI API key configured. Add OPENAI_API_KEY in Streamlit Secrets.")
    else:
        st.success("✅ OpenAI key loaded")

# ===================== Helper constants & functions =====================

# ---- Examples & schema enforce exam-style issues, not TOC ----
ISSUE_STYLE_EXAMPLES = """
Examples of acceptable 'issue' phrasing (Ontario):
- Appeal route from SCJ to Divisional Court — Civil
- Test for summary judgment — no genuine issue requiring a trial
- Limitation period — discoverability under s. 5, Limitations Act, 2002
- Motion to strike vs. motion for judgment on the pleadings — Rule 21 vs. Rule 25
- Rule 21 motion — striking claims with no reasonable cause of action
- Injunctive relief — RJR-MacDonald test
- Anton Piller order — elements and safeguards
- Mareva injunction — test and disclosure duties
- Standard of review on appeal — palpable and overriding error vs. correctness
- Security for costs — entitlement and factors (R. 56)
- Summary conviction appeal route — criminal
- Fiduciary duty — when it arises and remedies
"""

ISSUE_SCHEMA = """Return ONLY valid JSON with this shape:
{
  "issues": [
    {
      "issue": "short exam-style issue statement (must contain a specific decision/test/route/duty/remedy; not a chapter heading)",
      "triggers": ["2-5 short spotting cues from a fact pattern"],
      "rule_or_test": "concise numbered elements or test (Ontario)",
      "authorities": ["Rules/statutes/cases e.g., RCP r. 20.04; Hryniak 2014 SCC 7"],
      "pages": [int, ...]
    }
  ]
}"""

def _granularity_text(level: str) -> str:
    if level.startswith("Fine"):
        return ("Write concrete exam issues (route/test/standard/duty/remedy), not headings. "
                "Use the styles in the examples below.")
    if level.startswith("Medium"):
        return "Write practical sub-issues useful for exam answers; avoid broad chapter names."
    return "High level issues; fewer items."

def build_chunk_prompt(joined_pages: str, granularity: str, subject_hint: str, require_auth: bool) -> str:
    need_auth = ("Include specific Ontario/Canadian authorities (rules/statutes/cases) in 'authorities'."
                 if require_auth else
                 "Do not include citations unless essential.")
    return f"""You are building an exam index for the Ontario bar ({subject_hint}).

{_granularity_text(granularity)}
Avoid TOC-style headings like “Appeals” or “Pleadings”. Each 'issue' must read like an exam spotter, e.g., 'Appeal route from SCJ to Divisional Court — Civil'.
Prefer 'test/route/standard/duty/remedy' phrasing.

EXAMPLES:
{ISSUE_STYLE_EXAMPLES}

From the following pages, extract 12–25 issues using the JSON schema. {need_auth}

{ISSUE_SCHEMA}

Pages:
{joined_pages}
"""

def call_openai(messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int = 1400) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not initialized")
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def parse_json_issues(text: str) -> List[Dict[str, str]]:
    """Parse strict JSON; fallback: salvage bullets if JSON fails."""
    try:
        data = json.loads(text)
        items = data.get("issues", [])
        rows = []
        for it in items:
            rows.append({
                "issue": (it.get("issue") or "").strip(),
                "triggers": "; ".join(it.get("triggers") or [])[:300],
                "rule_or_test": (it.get("rule_or_test") or "").strip(),
                "authorities": "; ".join(it.get("authorities") or [])[:300],
                "pages": ", ".join(str(p) for p in (it.get("pages") or []))[:200],
                "notes": "",
            })
        return [r for r in rows if r["issue"]]
    except Exception:
        rows = []
        for line in text.splitlines():
            m = re.match(r"[-•]\s*(.+)", line.strip())
            if m:
                rows.append({"issue": m.group(1), "triggers": "", "rule_or_test": "", "authorities": "", "pages": "", "notes": ""})
        return rows

def looks_like_heading(s: str) -> bool:
    s0 = (s or "").strip()
    if not s0:
        return True
    if len(s0.split()) <= 2:
        return True
    cues = ["test", "standard", "duty", "route", "jurisdiction", "elements",
            "motion", "appeal", "remedy", "limitation", "discoverability",
            "burden", "on appeal", "leave", "service", "strike", "summary",
            "costs", "security"]
    return not any(c in s0.lower() for c in cues)

def refine_issue_statements(rows: List[Dict[str, str]], subject_hint: str, model: str, temperature: float) -> List[Dict[str, str]]:
    """Rewrite vague 'issue' strings into exam-style issue statements (batch)."""
    if client is None or not rows:
        return rows
    vague = [r for r in rows if looks_like_heading(r.get("issue",""))]
    if not vague:
        return rows
    bullet = "\n".join(f"- {r['issue']}" for r in vague)
    prompt = f"""Rewrite the following headings/topics into concise **exam-style issue statements**
for the Ontario bar ({subject_hint}). Use the patterns from these examples:
{ISSUE_STYLE_EXAMPLES}

Rules:
- Include a specific decision/test/route/duty/remedy in the phrasing.
- Keep it under 12 words.
- If the topic is 'Appeals', specify the appeal route or standard of review.
- If the topic is a remedy or motion, specify the test or rule (e.g., RCP).
Return one per line, in the same order, prefixed by '- ' and nothing else.

To rewrite:
{bullet}
"""
    try:
        out = call_openai(
            [{"role": "system", "content": "You rewrite headings into exam issue statements."},
             {"role": "user", "content": prompt}],
            model=model, temperature=temperature, max_tokens=600
        )
        new_lines = [ln.strip(" -\t") for ln in out.splitlines() if ln.strip()]
        j = 0
        for r in rows:
            if looks_like_heading(r.get("issue","")) and j < len(new_lines):
                r["issue"] = new_lines[j]
                j += 1
        return rows
    except Exception:
        return rows

def chunk_pages(pages: List[str], size: int) -> List[List[str]]:
    return [pages[i:i+size] for i in range(0, len(pages), size)]

# ===================== Sidebar =====================
with st.sidebar:
    st.header("Extraction style")
    subject_preset = st.selectbox(
        "Subject preset",
        ["Professional Responsibility", "Civil", "Criminal", "Family", "Public"],
        index=0,
    )
    # Keep a free-text override too (optional)
    subject_hint = st.text_input("Subject focus (optional override)", value=subject_preset)
    granularity = st.selectbox("Granularity", ["Fine (exam spotting)", "Medium", "Coarse"], index=0)
    require_authorities = st.checkbox("Include statutes/rules/cases", value=True)

# ===================== Session State =====================
for key, default in {
    "pdf_name": "",
    "pdf_bytes": b"",
    "pages_text": [],
    "issues_rows": [],
    "cheat_sheets_md": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ===================== Upload & Load (with diagnostics) =====================
st.markdown("### Upload bar materials (PDF)")

uploaded = st.file_uploader("Upload PDF (<= 200 MB)", type=["pdf"])
if uploaded is not None:
    st.session_state["pdf_name"] = uploaded.name
    st.session_state["pdf_bytes"] = uploaded.getvalue()

def extract_with_pymupdf(pdf_bytes: bytes, start: int, count: int) -> Tuple[list, list]:
    texts, errors = [], []
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        end = min(start + count, doc.page_count)
        for i in range(start, end):
            page = doc[i]
            try:
                txt = page.get_text("text") or ""
            except Exception as inner_e:
                txt = ""
                errors.append((i+1, f"PyMuPDF page error: {inner_e}"))
            texts.append((i+1, txt))
        doc.close()
    except Exception as e:
        errors.append(("open", f"PyMuPDF open error: {e}"))
    return texts, errors

def extract_with_pdfminer(pdf_bytes: bytes, page_numbers: list) -> dict:
    from pdfminer.high_level import extract_text
    tmp = io.BytesIO(pdf_bytes)
    out = {}
    for p in page_numbers:
        try:
            tmp.seek(0)
            out[p] = extract_text(tmp, page_numbers=[p-1]) or ""
        except Exception:
            out[p] = ""
    return out

if st.session_state["pdf_bytes"]:
    # Peek page count
    try:
        _doc = fitz.open(stream=st.session_state["pdf_bytes"], filetype="pdf")
        total_pages = _doc.page_count
        _doc.close()
        st.success(f"Loaded file: {st.session_state['pdf_name']} ({total_pages} pages)")
    except Exception as e:
        total_pages = None
        st.error(f"Could not open PDF: {e}")

    if total_pages:
        with st.expander("Load options", expanded=True):
            max_to_load = st.number_input(
                "How many pages to load (from page 1)",
                min_value=1, max_value=int(total_pages), value=min(50, int(total_pages)), step=1
            )
            do_load = st.button("Load PDF text")

        if do_load:
            st.session_state["pages_text"] = []
            try:
                prog = st.progress(0.0, text="Extracting text…")
                texts, errors = extract_with_pymupdf(st.session_state["pdf_bytes"], 0, int(max_to_load))

                empty_pages = [p for (p, t) in texts if not (t and t.strip())]
                if empty_pages:
                    st.info(f"{len(empty_pages)} page(s) had no selectable text; trying pdfminer fallback…")
                    pm_texts = extract_with_pdfminer(st.session_state["pdf_bytes"], empty_pages)
                    new_texts = []
                    for p, t in texts:
                        if (not t or not t.strip()) and p in pm_texts:
                            t = pm_texts[p]
                        new_texts.append((p, t))
                    texts = new_texts

                for i in range(len(texts)):
                    if i == len(texts) - 1 or (i + 1) % 5 == 0:
                        prog.progress((i + 1) / len(texts), text=f"Extracting… {i+1}/{len(texts)}")

                st.session_state["pages_text"] = [f"[Page {p}]\n{t}" for (p, t) in texts]
                if errors:
                    st.warning(f"PDF extraction warnings: {errors[:2]}{' ...' if len(errors)>2 else ''}")
                st.success(f"Pages loaded: {len(st.session_state['pages_text'])} / {total_pages}")
            except Exception as e:
                st.error(f"Error reading PDF: {e}")

# Preview
if st.session_state["pages_text"]:
    colA, colB = st.columns([1, 1])
    with colA:
        st.caption("Quick preview (first 3 loaded pages)")
        preview_pages = min(3, len(st.session_state["pages_text"]))
        preview = "\n\n".join(st.session_state["pages_text"][:preview_pages]).strip()
        st.text(preview if preview else "(No extractable text on these pages — they may be scanned images.)")
    with colB:
        st.metric("Pages loaded", len(st.session_state["pages_text"]))
        st.caption("Tip: start with 30–50 pages to verify, then increase.")

# ===================== Step 1: Generate Issues =====================
st.subheader("Step 1 • Generate exam-style issues (with page refs)")

if st.button("Generate issues"):
    if not st.session_state["pages_text"]:
        st.warning("Please load some PDF pages first.")
    elif client is None:
        st.warning("OpenAI client not configured.")
    else:
        with st.spinner("Extracting exam-style issues from chunks…"):
            # Chunk pages
            pages = st.session_state["pages_text"]
            chunks = chunk_pages(pages, int(chunk_size))
            chunk_rows_lists = []

            for idx, ch in enumerate(chunks, start=1):
                joined = "\n\n".join(ch)
                prompt = build_chunk_prompt(joined, granularity, subject_hint, require_authorities)
                try:
                    out = call_openai(
                        [
                            {"role": "system", "content": "You extract exam issues for the Ontario bar. Be precise and practical."},
                            {"role": "user", "content": prompt},
                        ],
                        model=model, temperature=temperature, max_tokens=1400,
                    )
                    rows = parse_json_issues(out)
                    if not rows:
                        # try stricter pass
                        out2 = call_openai(
                            [
                                {"role": "system", "content": "You return ONLY JSON."},
                                {"role": "user", "content": prompt + "\n\nReturn ONLY the JSON object, nothing else."},
                            ],
                            model=model, temperature=temperature, max_tokens=1400,
                        )
                        rows = parse_json_issues(out2)
                    if not rows:
                        st.warning(f"Chunk {idx}: no issues parsed.")
                    chunk_rows_lists.append(rows)
                except Exception as e:
                    st.error(f"OpenAI error on chunk {idx}: {e}")
                    st.stop()

            # Merge & dedupe
            all_rows = [r for lst in chunk_rows_lists for r in lst]

            def norm(s: str) -> str:
                return re.sub(r"\W+", " ", (s or "").lower()).strip()

            merged: Dict[str, Dict[str,str]] = {}
            for r in all_rows:
                key = norm(r["issue"])
                if not key:
                    continue
                if key not in merged:
                    merged[key] = r
                else:
                    if r.get("pages"):
                        merged[key]["pages"] = ", ".join(sorted(set((merged[key].get("pages","")+"," + r["pages"]).replace(" ", "").split(","))) - {""})
                    if r.get("authorities"):
                        merged[key]["authorities"] = "; ".join(sorted(set((merged[key].get("authorities","")+"; "+r["authorities"]).split("; "))) - {""})
                    if len(r.get("rule_or_test","")) > len(merged[key].get("rule_or_test","")):
                        merged[key]["rule_or_test"] = r["rule_or_test"]
                    if r.get("triggers"):
                        merged[key]["triggers"] = "; ".join(sorted(set((merged[key].get("triggers","")+"; "+r["triggers"]).split("; "))) - {""})

            final_rows = list(merged.values())[: int(max_issues)]
            # Refine vague headings to exam-style statements
            final_rows = refine_issue_statements(final_rows, subject_hint, model, temperature)

            st.session_state["issues_rows"] = final_rows
            st.success(f"Issues generated: {len(final_rows)}")

# ===================== Step 2: Review & Edit =====================
st.subheader("Step 2 • Review and edit the index")

issues_df = pd.DataFrame(
    st.session_state["issues_rows"] or [],
    columns=["issue", "triggers", "rule_or_test", "authorities", "pages", "notes"]
)

edited_df = st.data_editor(
    issues_df,
    num_rows="dynamic",
    use_container_width=True,
    key="issues_editor",
    column_config={
        "issue": st.column_config.TextColumn("Issue (exam-style)", width="medium"),
        "triggers": st.column_config.TextColumn("Spotting triggers", width="medium"),
        "rule_or_test": st.column_config.TextColumn("Rule / test (concise)", width="large"),
        "authorities": st.column_config.TextColumn("Authorities (rules/statutes/cases)", width="large"),
        "pages": st.column_config.TextColumn("Pages", width="small"),
        "notes": st.column_config.TextColumn("Your notes", width="large"),
    }
)

def dataframe_clean(df: pd.DataFrame) -> List[Dict[str, str]]:
    cleaned = []
    for _, row in df.iterrows():
        issue = str(row.get("issue", "")).strip()
        if not issue:
            continue
        cleaned.append({
            "issue": issue,
            "triggers": str(row.get("triggers", "")).strip(),
            "rule_or_test": str(row.get("rule_or_test", "")).strip(),
            "authorities": str(row.get("authorities", "")).strip(),
            "pages": str(row.get("pages", "")).strip(),
            "notes": str(row.get("notes", "")).strip(),
        })
    return cleaned

st.session_state["issues_rows"] = dataframe_clean(edited_df)

# Exports
colx, coly = st.columns(2)
with colx:
    if st.session_state["issues_rows"]:
        csv_bytes = pd.DataFrame(st.session_state["issues_rows"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download index as CSV", csv_bytes, file_name="bar_index.csv", mime="text/csv")
with coly:
    if st.session_state["issues_rows"]:
        md_lines = []
        for r in st.session_state["issues_rows"]:
            pages = f" (p. {r['pages']})" if r.get("pages") else ""
            auths = f" — {r['authorities']}" if r.get("authorities") else ""
            notes = f" — {r['notes']}" if r.get("notes") else ""
            md_lines.append(f"- {r['issue']}{pages}{auths}{notes}")
        md_text = "\n".join(md_lines)
        st.download_button("Download index as Markdown", md_text, file_name="bar_index.md", mime="text/markdown")

# ===================== Step 3: Cheat Sheets =====================
st.subheader("Step 3 • Generate cheat sheets for issues")

issues = st.session_state.get("issues_rows", [])
issues_count = len(issues)

if issues_count == 0:
    st.info("Create or import an index first (Steps 1–2).")
else:
    max_allowed = min(50, issues_count)
    default_val = min(10, max_allowed)
    min_allowed = 1  # keep flexible for small lists

    cols = st.columns([1, 1, 2])
    with cols[0]:
        top_k = st.number_input(
            "Number of issues to include",
            min_value=min_allowed, max_value=max_allowed, value=default_val, step=1,
        )
    with cols[1]:
        include_cases = st.checkbox("Include Canadian / Ontario cases", value=True)
    with cols[2]:
        st.caption("Cheat sheets: definition, elements/test, pitfalls, tiny checklist (and key cases if enabled).")

    if st.button("Generate cheat sheets"):
        if client is None:
            st.warning("OpenAI client not configured.")
        else:
            with st.spinner("Writing cheat sheets…"):
                selected = issues[: int(top_k)]
                issues_block = "\n".join(
                    f"- {r['issue']}"
                    + (f"\n  Rule/Test: {r['rule_or_test']}" if r.get('rule_or_test') else "")
                    + (f"\n  Authorities: {r['authorities']}" if r.get('authorities') else "")
                    + (f"\n  Pages: {r['pages']}" if r.get('pages') else "")
                    for r in selected
                )
                cases_hint = (
                    "Include leading Canadian and Ontario cases where relevant with very short parentheticals."
                    if include_cases else
                    "Do not include case citations. Focus on elements and tests."
                )
                cheat_prompt = f"""Create concise bar exam cheat sheets for the following issues.
Use this structure for each issue:
Issue
Definition (one–two lines)
Elements or test (numbered)
Common pitfalls or exceptions
Very short checklist for spotting/answering
{cases_hint}

Issues:
{issues_block}

Keep each issue under 120 words.
Return GitHub-flavored Markdown with ### headings for each issue.
"""
                try:
                    md = call_openai(
                        [
                            {"role": "system", "content": "You produce compact, accurate bar exam cheat sheets."},
                            {"role": "user", "content": cheat_prompt},
                        ],
                        model=model, temperature=temperature, max_tokens=1800,
                    )
                    st.session_state["cheat_sheets_md"] = md
                    st.success("Cheat sheets created")
                except Exception as e:
                    st.error(f"OpenAI error while creating cheat sheets: {e}")

if st.session_state.get("cheat_sheets_md"):
    st.markdown("### Cheat sheets (editable)")
    cheat_md = st.text_area("Markdown", st.session_state["cheat_sheets_md"], height=500, key="cheat_md_area")
    st.session_state["cheat_sheets_md"] = cheat_md
    st.download_button(
        "Download cheat sheets as Markdown",
        st.session_state["cheat_sheets_md"],
        file_name="cheat_sheets.md",
        mime="text/markdown",
    )
