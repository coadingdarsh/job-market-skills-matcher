# dashboard/app.py

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pypdf import PdfReader
import docx


# -----------------------------
# Helpers
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
    s = s.strip().lower()
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
    gaps = sorted(list(j - r))
    return gaps


# -----------------------------
# REAL Parser: uses uploaded resume file
# -----------------------------
def run_your_parser(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Reads the uploaded resume file and returns extracted text + basic metadata.
    This replaces the demo output so every upload reflects the current resume.
    """
    suffix = Path(filename).suffix.lower() or ".pdf"

    # Save upload to a temp file so PDF/DOCX libs can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Extract text based on file type
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

    text = (text or "").strip()

    # Return REAL extracted content (no more demo)
    return {
        "candidate": {"name": filename, "email": "", "phone": ""},
        "summary": (text[:400] + "…") if len(text) > 400 else text,
        "skills": [],  # Optional: we can add skills extraction next
        "experience": [],
        "education": [],
        "projects": [],
        "raw_text": text,
    }


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="AI Resume Parser Dashboard", layout="wide")

st.title("📄 Skills Gap Finder ")
st.caption("Upload a resume, inspect extracted fields, and test role matching.")

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader("Resume file", type=["pdf", "docx", "txt"])
    st.divider()
    st.header("Job Matching")
    st.write("Add one or more target roles and their required skills.")
    default_jobs = [
        {"role": "Business Analyst Intern", "skills": "Excel, SQL, Requirements Gathering, Stakeholder Management, Power BI"},
        {"role": "Data Science Co-op", "skills": "Python, Statistics, Machine Learning, SQL, Pandas, Visualization"},
        {"role": "Consulting Intern", "skills": "Problem Solving, Communication, PowerPoint, Excel, Stakeholder Management"},
    ]
    jobs = st.session_state.get("jobs", default_jobs)

    if st.button("Reset sample roles"):
        st.session_state["jobs"] = default_jobs
        jobs = default_jobs

    # editable jobs
    edited = []
    for i, j in enumerate(jobs):
        st.subheader(f"Role {i+1}")
        role = st.text_input(f"Role name {i+1}", j["role"], key=f"role_{i}")
        skills = st.text_area(f"Skills (comma-separated) {i+1}", j["skills"], height=70, key=f"skills_{i}")
        edited.append({"role": role, "skills": skills})
    st.session_state["jobs"] = edited

col1, col2 = st.columns([1.2, 1])

if not uploaded:
    st.info("Upload a resume to see the parsed output and dashboard analytics.")
    st.stop()

file_bytes = uploaded.getvalue()
parsed = run_your_parser(file_bytes, uploaded.name)

# Top summary cards
cand_name = safe_get(parsed, ["candidate", "name"], "Unknown")
cand_email = safe_get(parsed, ["candidate", "email"], "—")
cand_phone = safe_get(parsed, ["candidate", "phone"], "—")
skills = parsed.get("skills", []) or []

with col1:
    st.subheader("✅ Extracted Candidate Profile")
    a, b, c = st.columns(3)
    a.metric("Name", cand_name)
    b.metric("Email", cand_email)
    c.metric("Phone", cand_phone)

    st.write("**Summary (from uploaded resume text)**")
    st.write(parsed.get("summary", "—") or "—")

    st.write("**Skills**")
    if skills:
        st.write(" ".join([f"`{s}`" for s in skills]))
    else:
        st.warning("No skills extracted yet (skills extraction can be added next).")

    st.divider()
    st.subheader("🧩 Sections")
    tabs = st.tabs(["Experience", "Education", "Projects", "Raw Text", "Raw JSON"])

    with tabs[0]:
        exp = parsed.get("experience", []) or []
        if not exp:
            st.write("—")
        for e in exp:
            st.markdown(f"**{e.get('title','')} — {e.get('company','')}**")
            for bullet in e.get("bullets", []) or []:
                st.write(f"• {bullet}")

    with tabs[1]:
        edu = parsed.get("education", []) or []
        if not edu:
            st.write("—")
        for ed in edu:
            st.markdown(f"**{ed.get('school','')}** — {ed.get('degree','')}")

    with tabs[2]:
        projs = parsed.get("projects", []) or []
        if not projs:
            st.write("—")
        for p in projs:
            st.markdown(f"**{p.get('name','')}**")
            st.write(p.get("desc", ""))

    with tabs[3]:
        st.text_area("Extracted resume text", parsed.get("raw_text", ""), height=380)

    with tabs[4]:
        st.json(parsed)

with col2:
    st.subheader("🎯 Job Match Scores")

    job_rows = []
    for j in st.session_state["jobs"]:
        role = j["role"].strip()
        job_skills = [s.strip() for s in j["skills"].split(",") if s.strip()]
        s = score_match(skills, job_skills)
        gaps = skill_gaps(skills, job_skills)
        job_rows.append(
            {
                "Role": role,
                "Match %": s,
                "Missing skills": ", ".join(gaps[:10]) + ("..." if len(gaps) > 10 else ""),
            }
        )

    df = pd.DataFrame(job_rows).sort_values("Match %", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.write("**Top Role Fit:**", df.iloc[0]["Role"] if len(df) else "—")

    st.bar_chart(df.set_index("Role")["Match %"])

    st.divider()
    st.subheader("⬇️ Export")
    st.download_button(
        "Download parsed JSON",
        data=json.dumps(parsed, indent=2).encode("utf-8"),
        file_name="parsed_resume.json",
        mime="application/json",
    )
