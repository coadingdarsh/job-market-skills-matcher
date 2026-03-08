Job Market Skills Matcher (AI Job Recommender)

Live Demo:
https://skillsgap-ai.streamlit.app/

An AI-powered job recommendation system built using real LinkedIn job postings to help users discover career opportunities that align with their skills. The project combines large-scale job market analysis and natural language processing (NLP) to recommend jobs based on skill similarity rather than job titles.

Traditional job search platforms rely heavily on keyword or title matching. This system focuses on skills-first matching, helping users identify roles that align with their actual capabilities.

Problem

Many professionals entering a new job market face a skills translation gap.

Even when candidates possess relevant experience, job postings often describe required skills using different terminology across industries, companies, and regions.

Example:

Candidate Skill	Job Posting Skill
Performance Marketing	Growth Marketing
SQL Analysis	Data Analytics
Campaign Optimization	Marketing Analytics

As a result, qualified candidates may miss opportunities because their experience is described differently than job postings.

This project aims to bridge that gap using NLP-based similarity matching.

Personal Motivation

When I moved to Canada, I experienced this challenge firsthand.

While working in performance marketing in Dubai, many of my skills aligned with roles in Canada, but job postings used different terminology to describe similar work.

This project explores how data-driven skill matching and job market analysis can help professionals discover opportunities that better reflect their real capabilities.

Project Objectives

This project focuses on two main goals:

1. Job Market Analysis

Analyze a large dataset of LinkedIn job postings to identify trends in the modern job market, including:

Most in-demand job roles

Hiring trends across industries

Common skill requirements across positions

Geographic job distribution

2. AI Job Recommendation System

Develop a content-based recommender system that suggests job opportunities based on a user’s skills.

The system compares user skills with job posting requirements and returns ranked job recommendations with similarity scores.

Dataset

The project uses a large public dataset of LinkedIn job postings.

Dataset:
1.3M+ LinkedIn Jobs and Skills (2024)

Source: Kaggle

The dataset contains millions of job records with both structured and unstructured information.

Key fields used:

Field	Description
job_title	Raw job posting title
company_name	Hiring company
location	Job location
job_skills	Skills extracted from job descriptions

The dataset was cleaned and processed to support both job market analysis and skill-based recommendation.

Data Cleaning & Normalization

Real-world job data is often messy and inconsistent.

The preprocessing pipeline performs several steps to prepare the data for analysis and modeling.

Key preprocessing steps:

Removing duplicate job entries

Standardizing company names and locations

Cleaning and normalizing skill text

Removing punctuation and irrelevant tokens

Converting text to lowercase

Removing stopwords

Regular expressions were also used to normalize job titles and categorize roles into standardized groups, improving analysis accuracy.

Exploratory Data Analysis (EDA)

Exploratory analysis was conducted to understand global job market patterns.

Insights explored include:

Most common job roles

Top hiring companies

Geographic distribution of jobs

Skill demand across industries

Since the dataset contains a large proportion of U.S. postings, comparisons were normalized using percentage-based metrics rather than raw counts to ensure fair comparisons across countries.

System Architecture

The recommendation engine follows a content-based recommendation pipeline.

User Skills → Text Processing → Feature Engineering → Similarity Calculation → Ranked Job Recommendations

Recommendation Pipeline
1. Skill Text Processing

User skills and job skills are cleaned and standardized using NLP preprocessing techniques:

Lowercase conversion

Punctuation removal

Stopword removal

Tokenization

2. Feature Engineering

Skill text is transformed into numerical vectors using:

TF-IDF (Term Frequency – Inverse Document Frequency)

TF-IDF identifies important skill terms across thousands of job postings and represents them as numerical feature vectors.

3. Similarity Calculation

User skill vectors are compared with job skill vectors using:

Cosine Similarity

This produces a match score between 0 and 1, indicating how closely a candidate’s skills align with job requirements.

4. Job Ranking

Jobs are ranked based on similarity score and returned as Top-K recommendations.

Optional filters allow users to narrow results by:

location

experience level

employment type

Example Output

The recommender system returns ranked job recommendations with similarity scores.

Rank	Job Title	Match Score
1	Marketing Data Analyst	0.86
2	Growth Marketing Specialist	0.79
3	Digital Marketing Analyst	0.75

The similarity score provides transparent and explainable matching results.

Interactive Application

The recommendation system is deployed as an interactive web application using Streamlit.

Users can:

Input their skills

Apply filters such as location or job type

View ranked job recommendations

Explore similarity scores visually

Live Demo:
https://skillsgap-ai.streamlit.app/

Technology Stack

Programming Language

Python

Data Processing

Pandas

NumPy

Natural Language Processing

NLTK

Regular Expressions

Machine Learning

Scikit-learn

TF-IDF Vectorization

Cosine Similarity

Visualization

Matplotlib

Seaborn

Plotly

Application

Streamlit

Project Benefits
For Job Seekers

Helps identify roles aligned with existing skills

Improves job discovery beyond keyword search

Provides transparent similarity-based recommendations

For Newcomers & Career Switchers

Translates previous experience into local job market terminology

Reduces the skills translation gap

Supports career discovery through skill-based matching

For Data Analysis

Demonstrates large-scale job market analytics

Uses real-world job data to identify hiring trends

Future Improvements

Potential future improvements include:

Sentence embeddings using BERT or SBERT

Skill ontology mapping

Hybrid recommender systems

Real-time job data integration through APIs

Career pathway recommendation features

Why This Project Matters

Many job search platforms rely primarily on title-based matching, which can fail for newcomers, interdisciplinary professionals, and career switchers.

A skills-first recommendation approach provides a more accurate way to connect people with opportunities that align with their capabilities.

This project demonstrates how job market analytics, natural language processing, and recommender systems can help improve the job discovery process.
