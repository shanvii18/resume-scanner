# Resume Scanner

## Overview
Resume Scanner is an AI-powered tool designed to make evaluating resumes faster and smarter. It helps job seekers understand how recruiters view their resumes and assists recruiters in quickly shortlisting candidates. The goal is to save time, highlight key skills, and provide actionable insights for improvement.

## Key Features
- Automatic Resume Parsing: Extracts education, skills, and experience from resumes effortlessly.
- Skill Analysis: Identifies technical and soft skills in a candidate’s resume.
- Ranking & Suggestions: Provides a score and practical recommendations to improve resumes.
- Simple Web Interface: Streamlit-based app for easy resume upload and analysis.

## Technology Used
- **Language:** Python  
- **Libraries:** Streamlit, scikit-learn, pandas, numpy, plotly  
- **Machine Learning Techniques:** TF-IDF Vectorization, Cosine Similarity, Random Forest Classifier  
- **Data Storage:** Pickle (for saving trained models)

## How to Run
1. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Start the app**:
   ```bash
   streamlit run app.py
4. **Open your browser to see the app and upload resumes to analyze**.


## Project Structure
resume-scanner/
│
├─ app.py               # Main Streamlit application
├─ model.py             # Machine learning model
├─ data/                # Sample datasets
├─ utils/               # Helper functions
├─ requirements.txt     # Python dependencies
└─ README.md            # This file

## Screenshots

### Dashboard
The main dashboard provides a clean interface where users can upload resumes and view overall analysis.  
#### 1. Company + Job Title Mode
This view allows users to analyze resumes against a specific company and job title. It highlights relevant skills, experience, and gaps for targeted applications.  
<img width="1344" height="599" alt="Screenshot 2026-03-30 181159" src="https://github.com/user-attachments/assets/905aa2bd-6bc3-4dd1-93ba-095f92a4ba68" />

---

### 2. Job Description Paste Mode
Users can paste a job description here, and the app evaluates resumes against the JD. It shows matching skills, missing keywords, and an overall score for optimization.  
<img width="1346" height="591" alt="Screenshot 2026-03-30 181454" src="https://github.com/user-attachments/assets/c6e9e170-d849-45a8-87b3-45c3500aa72c" />


---

### 4. Results
After analysis, the results page summarizes the resume evaluation:
- Skill match percentage  
- Missing or weak areas  
- Suggestions for improvement(a feature to be integrated soon)
<img width="1041" height="554" alt="Screenshot 2026-03-30 181221" src="https://github.com/user-attachments/assets/c242fd25-2a31-435f-81d7-22a307c24830" />


## Future Plans
- Add support for more file types like DOCX and TXT.
- Improve AI suggestions using advanced NLP techniques.
- Integrate with job portals for automated application tips.

  
##Author

Shanvi Verma







   
