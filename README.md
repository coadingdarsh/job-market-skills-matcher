# Job Market Skills Matcher (AI Job Recommender)

An end-to-end **skills-based job recommender system** built on real LinkedIn job postings.

**Problem this solves:** when you're new to a country or job market, you often have strong experience—but the *local job postings describe skills differently*. This project reduces that mismatch by translating a user’s skills into the “posting language”, then ranking jobs by similarity.

> Personal motivation: I faced this skills-translation gap when I first moved to Canada. Back home in Dubai I ran a results-driven ads business for clients, but breaking into a new market meant learning how employers describe requirements locally. This system is designed to help newcomers bridge that same gap.

---

## What it does

Given:
- a **jobs** dataset with a `job_skills` field (plus optional metadata like location, level, type)
- a **user queries** file with a `job_skills` field

It:
1. Cleans + normalizes skill text (lowercase, remove non-letters, remove stopwords)
2. Builds a **TF–IDF** representation for job skills
3. Computes **cosine similarity** between a user’s skills and job skills
4. Applies optional filters (city, country, level, type)
5. Outputs **Top‑K ranked matches** with a similarity score

---

## Repo structure

```
.
├── job_recommender_system.py
├── data/
│   └── test_cases.csv
├── outputs/
│   └── output_job_skills_match.csv
├── EDA.ipynb
├── data_cleaning.ipynb
├── stopwords.txt
└── LICENSE
```

---

## Quickstart

### 1) Create a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the recommender

```bash
python job_recommender_system.py \
  --jobs data/linkedin_job_posts_skills.csv \
  --queries data/test_cases.csv \
  --out outputs/output_job_skills_match.csv \
  --topk 5
```

> Note: the large LinkedIn dataset is not included in the repo. See **Data** below.

---

## Data

This project was developed using:

- **Dataset:** *1.3M+ LinkedIn Jobs and Skills (2024)*
- **Source:** Kaggle (asaniczka)

Because the raw dataset is large, the repo contains only:
- `data/test_cases.csv` (sample user inputs)
- `outputs/output_job_skills_match.csv` (example output)

---

## Output format

The system writes a CSV with columns like:
- `query_sn` (query id)
- `users_skills` (raw user skills)
- `job_title`, `company`, `job_location`
- `job_skills` (skills for the job posting)
- `job_suitability_rank` (1…K)
- `similarity_score` (0…1)

---

## Why this stands out (for employers)

This isn’t “just a notebook” — it’s a **reproducible pipeline** that shows:

- **Product thinking:** clear inputs → clear outputs, with filtering and ranking.
- **Applied NLP:** TF‑IDF + cosine similarity for explainable matching.
- **Real-world framing:** focuses on the *skills-translation problem* for newcomers.
- **Extensibility:** easy path to upgrade to embeddings, skill ontology mapping, and a web UI.

If you’re reviewing this repo, check out:
- `job_recommender_system.py` for the matching + ranking pipeline
- `data_cleaning.ipynb` for how the dataset is prepared
- `EDA.ipynb` for market insights and role-demand analysis

---

## Roadmap (next iterations)

- skill normalization improvements (synonyms: “analytics” vs “analytical skills”)
- Hybrid matching: TF‑IDF + embeddings (e.g., Sentence Transformers)
- Explainability: “missing skills” per recommended job
- Simple web app (Streamlit/FastAPI) with a clean UI

---

## License

This project is released under the license in `LICENSE`.
