# dashboard/app.py

import json
import re
import tempfile
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd

from pypdf import PdfReader
import docx


# -----------------------------
# General helpers
# -----------------------------
def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return default
    return cur


def normalize_skill(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def score_match(resume_skills: List[str], job_skills: List[str]) -> float:
    r = set(map(normalize_skill, resume_skills))
    j = set(map(normalize_skill, job_skills))
    if not j:
        return 0.0
    return round(100 * (len(r & j) / len(j)), 2)


def skill_gaps(resume_skills: List[str], job_skills: List[str]) -> List[str]:
    r = set(map(normalize_skill, resume_skills))
    j = set(map(normalize_skill, job_skills))
    return sorted(list(j - r))


# -----------------------------
# Resume text extraction
# -----------------------------
def extract_text_from_upload(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix if suffix else ".tmp") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    text = ""
    if suffix == ".pdf":
        reader = PdfReader(tmp_path)
        text = "\n".join([(p.extract_text() or "") for p in reader.pages])
    elif suffix == ".docx":
        d = docx.Document(tmp_path)
        text = "\n".join(p.text for p in d.paragraphs)
    else:
        # txt fallback
        text = file_bytes.decode("utf-8", errors="ignore")

    return (text or "").strip()


def normalize_resume_text(text: str) -> str:
    # keep letters/numbers/+ and spaces; collapse whitespace
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9+\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# -----------------------------
# Skill vocab + extraction
# -----------------------------
@st.cache_data(show_spinner=False)
def load_skill_vocabulary() -> Dict[str, Any]:
    """
    Build a vocabulary from CSVs in the repo that include 'job_skills'.
    Uses files that exist in your repo root, so it works on Streamlit Cloud.
    """
    repo_root = Path(__file__).resolve().parents[1]

    candidate_paths = [
        repo_root / "linkedin_job_postings.csv",
        repo_root / "linkedin_job_posts_skills.csv",
        repo_root / "test_cases.csv",  # ✅ exists in your repo
        repo_root / "output_job_skills_match.csv",
    ]

    chosen_path = None
    chosen_cols = None

    for p in candidate_paths:
        if p.exists():
            try:
                df_head = pd.read_csv(p, nrows=3)
                cols = df_head.columns.tolist()
                if "job_skills" in cols:
                    chosen_path = p
                    chosen_cols = cols
                    break
            except Exception:
                continue

    if chosen_path is None:
        return {
            "vocab": set(),
            "source_file": None,
            "error": "No CSV found with a 'job_skills' column in repo root.",
        }

    df = pd.read_csv(chosen_path)
    vocab = set()

    for cell in df["job_skills"].dropna().astype(str):
        for s in cell.split(","):
            s = s.strip().lower()
            if 2 <= len(s) <= 60:
                vocab.add(s)

    return {
        "vocab": vocab,
        "source_file": str(chosen_path.name),
        "error": None,
        "columns": chosen_cols,
        "vocab_size": len(vocab),
    }


def extract_skills_from_text(resume_text: str, vocab: set[str], max_skills: int = 50) -> List[str]:
    """
    Match known skills (from vocab) inside resume text.
    Uses boundary-ish regex and matches longer skills first.
    """
    if not resume_text or not vocab:
        return []

    t = normalize_resume_text(resume_text)

    found = []
    for skill in sorted(vocab, key=len, reverse=True):
        s = normalize_resume_text(skill)
        if not s:
            continue

        pattern = r"(^|\s)" + re.escape(s) + r"(\s|$)"
        if re.search(pattern, t):
            found.append(skill)
        if len(found) >= max_skills:
            break

    # Nicely format for UI
    return [x.title() for x in found]


# -----------------------------
# Main "parser" output (for dashboard)
# -----------------------------
def run_your_parser(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    text = extract_text_from_upload(file_bytes, filename)

    vocab_info = load_skill_vocabulary()
    vocab = vocab_info.get("vocab", set()) or set()
    skills = extract_skills_from_text(text, vocab, max_skills=50)

    return {
        "candidate": {"name": filename, "email": "", "phone": ""},
        "summary": (text[:500] + "…") if len(text) > 500 else text,
        "skills": skills,
        "raw_text": text,
        "vocab_meta": {
            "source_file": vocab_info.get("source_file"),
            "vocab_size": vocab_info.get("vocab_size", 0),
            "error": vocab_info.get("error"),
        },
    }


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Skills Gap Finder", layout="wide")

st.title("Talent Skills Gap Analyzer")
st.caption("Upload a resume → extract skills → compare against target roles → identify gaps.")

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Resume file", type=["pdf", "docx", "txt"])

    st.divider()
    st.header("Target Roles")

    default_jobs = [
        {"role": "Business Analyst Intern", "skills": "Excel, SQL, Requirements Gathering, Stakeholder Management, Power BI"},
        {"role": "Data Science Co-op", "skills": "Python, Statistics, Machine Learning, SQL, Pandas, Visualization"},
        {"role": "Consulting Intern", "skills": "Problem Solving, Communication, PowerPoint, Excel, Stakeholder Management"},
    ]
    jobs = st.session_state.get("jobs", default_jobs)

    if st.button("Reset sample roles"):
        st.session_state["jobs"] = default_jobs
        jobs = default_jobs

    edited = []
    for i, j in enumerate(jobs):
        st.subheader(f"Role {i+1}")
        role = st.text_input(f"Role name {i+1}", j["role"], key=f"role_{i}")
        skills_txt = st.text_area(
            f"Skills (comma-separated) {i+1}",
            j["skills"],
            height=70,
            key=f"skills_{i}",
        )
        edited.append({"role": role, "skills": skills_txt})
    st.session_state["jobs"] = edited

    st.divider()
    show_debug = st.checkbox("Show debug info", value=False)


col1, col2 = st.columns([1.25, 1])

if not uploaded:
    st.info("Upload a resume to generate skill extraction + role match results.")
    st.stop()

file_bytes = uploaded.getvalue()
parsed = run_your_parser(file_bytes, uploaded.name)

skills = parsed.get("skills", []) or []
resume_text = parsed.get("raw_text", "") or ""

# Left column: extracted info
with col1:
    st.subheader("✅ Extracted Resume Info")

    a, b = st.columns([1, 2])
    a.metric("File", safe_get(parsed, ["candidate", "name"], "—"))
    b.metric("Skills Found", str(len(skills)))

    st.write("**Summary (from resume text)**")
    st.write(parsed.get("summary", "—") or "—")

    st.write("**Extracted Skills**")
    if skills:
        st.write(" ".join([f"`{s}`" for s in skills]))
    else:
        st.warning("No skills found. This usually means your skill vocabulary is empty or too small for your resume.")

    tabs = st.tabs(["Raw Text", "Export JSON"])
    with tabs[0]:
        st.text_area("Extracted resume text", resume_text, height=360)

    with tabs[1]:
        st.download_button(
            "Download parsed JSON",
            data=json.dumps(parsed, indent=2).encode("utf-8"),
            file_name="parsed_resume.json",
            mime="application/json",
        )

    if show_debug:
        st.divider()
        st.subheader("🛠 Debug")
        vocab_meta = parsed.get("vocab_meta", {})
        st.write("Vocabulary source file:", vocab_meta.get("source_file"))
        st.write("Vocabulary size:", vocab_meta.get("vocab_size"))
        if vocab_meta.get("error"):
            st.error(vocab_meta.get("error"))

# Right column: job matching
with col2:
    st.subheader("🎯 Role Match (Skills-based)")

    if not st.session_state["jobs"]:
        st.warning("No roles configured.")
        st.stop()

    job_rows = []
    for j in st.session_state["jobs"]:
        role = (j.get("role") or "").strip() or "Untitled Role"
        job_skills = [s.strip() for s in (j.get("skills") or "").split(",") if s.strip()]

        match = score_match(skills, job_skills)
        gaps = skill_gaps(skills, job_skills)

        job_rows.append(
            {
                "Role": role,
                "Match %": match,
                "Missing skills": ", ".join(gaps[:12]) + ("..." if len(gaps) > 12 else ""),
                "_missing_count": len(gaps),
            }
        )

    df = pd.DataFrame(job_rows).sort_values(["Match %", "_missing_count"], ascending=[False, True]).drop(columns=["_missing_count"])

    # Consulting-style highlight
    if len(df):
        st.success(f"Top Recommended Role: **{df.iloc[0]['Role']}**  —  **{df.iloc[0]['Match %']}%** match")

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.bar_chart(df.set_index("Role")["Match %"])

    if len(df) and df.iloc[0]["Match %"] == 0 and len(skills) == 0:
        st.info("Tip: Skills extraction is empty → match scores will be 0. Check Debug for vocab size/source.")
