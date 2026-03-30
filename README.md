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



## Future Plans
- Add support for more file types like DOCX and TXT.
- Improve AI suggestions using advanced NLP techniques.
- Integrate with job portals for automated application tips.

  
##Author

Shanvi Verma







   
