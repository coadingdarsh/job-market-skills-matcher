# Job Market Skills Matcher (AI Job Recommender)

**Live Demo:** https://skillsgap-ai.streamlit.app/

An end-to-end **skills-based job recommender** built on real LinkedIn job postings to solve the **skills-translation gap** for newcomers and career switchers.

**Problem:** You may have strong experience, but employers describe required skills differently in a new market.  
**Solution:** This system normalizes skill text, translates it into “job-posting language,” and ranks jobs by similarity with **explainable matching (TF-IDF + cosine similarity).**

---

## Personal motivation

When I moved to Canada, I faced this skills translation gap firsthand. Back home in Dubai, I ran performance marketing campaigns for clients, but breaking into a new market meant learning how job postings describe the same skills differently.  
This project helps newcomers reduce that mismatch and find roles that match their real capabilities.

---

## What it does

**Inputs**
- Jobs dataset with a `job_skills` field (+ optional metadata: location, level, type)
- User queries file with a `job_skills` field

**Pipeline**
1. Clean + normalize skill text (lowercase, remove noise, remove stopwords)
2. Vectorize job skills with **TF-IDF**
3. Compute **cosine similarity** between user skills and job skills
4. Apply optional filters (city, country, level, type)
5. Output **Top-K ranked matches** with a similarity score


**App UI**
<img width="952" height="439" alt="dashboard for resume parser" src="https://github.com/user-attachments/assets/7cb23eaa-e2d4-4eff-972c-3de1f44494cd" />





