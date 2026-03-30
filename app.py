import streamlit as st
import pymupdf
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
import json

# PAGE CONFIG
st.set_page_config(
    page_title="HireIQ - Resume Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8;
}

.hero-header {
    background: linear-gradient(135deg, #0f1629 0%, #1a2744 50%, #0f1629 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    line-height: 1.2;
}
.hero-title span {
    color: #38bdf8;
}
.hero-subtitle {
    font-size: 1rem;
    color: #64748b;
    margin-top: 0.5rem;
    font-weight: 400;
    letter-spacing: 0.02em;
}
.hero-badge {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.1em;
}

.card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.card:hover {
    border-color: #2d4a7a;
}
.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #38bdf8;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}

.metric-card {
    background: #0f1629;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 1.25rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #38bdf8, transparent);
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 0.25rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.score-high { color: #22c55e; }
.score-mid  { color: #f59e0b; }
.score-low  { color: #ef4444; }

.skill-tag {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 6px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin: 0.2rem;
    font-weight: 400;
}
.skill-found {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    color: #4ade80;
}
.skill-missing {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #f87171;
}

.section-divider {
    border: none;
    border-top: 1px solid #1e2d4a;
    margin: 2rem 0;
}

[data-testid="stFileUploader"] {
    background: #0f1629;
    border: 2px dashed #1e3a5f !important;
    border-radius: 12px;
    padding: 1rem;
}
[data-testid="stFileUploader"]:hover {
    border-color: #38bdf8 !important;
}

.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #38bdf8);
    color: #0a0e1a;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    padding: 0.6rem 1.5rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(56,189,248,0.3);
}

.stSelectbox > div > div,
.stTextArea > div > div {
    background: #0f1629 !important;
    border-color: #1e2d4a !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

.stRadio > div {
    gap: 1rem;
}
.stRadio label {
    color: #94a3b8 !important;
}

.streamlit-expanderHeader {
    background: #0f1629 !important;
    border: 1px solid #1e2d4a !important;
    border-radius: 8px !important;
    color: #94a3b8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 2px; }

.stTabs [data-baseweb="tab-list"] {
    background: #0f1629;
    border-radius: 10px;
    padding: 0.3rem;
    gap: 0.25rem;
    border: 1px solid #1e2d4a;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px;
    color: #64748b !important;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #38bdf8 !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #0ea5e9, #38bdf8) !important;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


# JSON LOADING
with open("roles.json", "r") as f:
    roles_data = json.load(f)

with open("companies.json", "r") as f:
    companies_data = json.load(f)

master_skills = set()
for role in roles_data:
    for skill in roles_data[role]:
        master_skills.add(skill.lower())
for company in companies_data:
    for role in companies_data[company]:
        for skill in companies_data[company][role]:
            master_skills.add(skill.lower())
master_skills = list(master_skills)

lemmatizer = WordNetLemmatizer()


# PREPROCESSING
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9. ]', '', text)
    return text.strip()

def remove_stopwords(text):
    words = text.split()
    return " ".join([w for w in words if w not in ENGLISH_STOP_WORDS])

def lemmatize_text(text):
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words])

def extract_text(uploaded_file):
    ext = uploaded_file.name.split(".")[-1].lower()
    text = ""
    if ext == "pdf":
        file_bytes = uploaded_file.read()
        doc = pymupdf.open(stream=file_bytes, filetype="pdf")
        for page in doc:
            text += page.get_text()
    else:
        st.error("Unsupported file type")
    return text

def get_required_skills(company, role):
    company = company.lower()
    role = role.lower()
    if company in companies_data and role in companies_data[company]:
        return companies_data[company][role]
    if role in roles_data:
        return roles_data[role]
    return {}

def weighted_match(resume_text, required_skills):
    total_weight = sum(required_skills.values())
    matched_weight = 0
    found_skills, missing_skills = [], []
    for skill, weight in required_skills.items():
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, resume_text):
            matched_weight += weight
            found_skills.append(skill)
        else:
            missing_skills.append(skill)
    if total_weight == 0:
        return [], [], 0
    match_percentage = (matched_weight / total_weight) * 100
    return found_skills, missing_skills, round(match_percentage, 2)


# TF-IDF COSINE SIMILARITY
def get_cosine_similarity(resume_text, jd_text):
    if not jd_text or not jd_text.strip():
        return 0
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)

def get_final_score(match_percent, cosine_score, has_jd=False):
    if has_jd:
        return round((match_percent * 0.6) + (cosine_score * 0.4), 2)
    return match_percent


# HELPERS
def get_score_class(score):
    if score >= 70:
        return "score-high"
    elif score >= 40:
        return "score-mid"
    return "score-low"

def plot_skill_gap(required_skills, found_skills):
    skills = list(required_skills.keys())
    weights = [required_skills[s] for s in skills]
    achieved = [required_skills[s] if s in found_skills else 0 for s in skills]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0f1629')
    ax.set_facecolor('#0a0e1a')

    x = range(len(skills))
    bar_w = 0.35

    ax.bar([i - bar_w/2 for i in x], weights, bar_w,
           color='#1e3a5f', label='Required', zorder=2)
    ax.bar([i + bar_w/2 for i in x], achieved, bar_w,
           color='#38bdf8', label='Matched', zorder=2)

    ax.set_xticks(list(x))
    ax.set_xticklabels(skills, rotation=40, ha='right',
                       color='#94a3b8', fontsize=9)
    ax.set_ylabel("Weight", color='#64748b', fontsize=9)
    ax.set_title("Skill Weight Analysis", color='#e2e8f0',
                 fontsize=11, fontweight='bold', pad=15)
    ax.tick_params(colors='#64748b')
    ax.spines['bottom'].set_color('#1e2d4a')
    ax.spines['left'].set_color('#1e2d4a')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.label.set_color('#64748b')
    ax.grid(axis='y', color='#1e2d4a', linewidth=0.5, zorder=0)
    ax.legend(facecolor='#0f1629', edgecolor='#1e2d4a',
              labelcolor='#94a3b8', fontsize=8)

    plt.tight_layout()
    return fig


# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0'>
        <div style='font-family: Space Mono, monospace; font-size: 1.1rem;
                    color: #38bdf8; font-weight: 700; margin-bottom: 0.25rem;'>
            HireIQ
        </div>
        <div style='font-size: 0.75rem; color: #475569; margin-bottom: 1.5rem;'>
            AI Resume Analyzer v0.1
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Match Mode")
    option = st.radio(
        "",
        ["Company + Role", "Paste JD"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border-color:#1e2d4a; margin: 1.5rem 0'>", unsafe_allow_html=True)

    if "Company" in option:
        st.markdown("### Select Target")
        company = st.selectbox("Company", list(companies_data.keys()))
        role = st.selectbox("Role", list(roles_data.keys()))
        raw_jd = None
    else:
        company, role = None, None
        st.markdown("### Job Description")
        raw_jd = st.text_area(
            "",
            placeholder="Paste the full job description here...",
            height=250,
            label_visibility="collapsed"
        )

    st.markdown("<hr style='border-color:#1e2d4a; margin: 1.5rem 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 0.7rem; color: #334155; line-height: 1.8;'>
        <div>PDF resumes only</div>
        <div>Multi-resume supported</div>
        <div>NLP + Weighted Matching + TF-IDF</div>
    </div>
    """, unsafe_allow_html=True)


# MAIN CONTENT
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">AI POWERED · NLP DRIVEN · TF-IDF SCORING</div>
    <div class="hero-title">Hire<span>IQ</span> Resume Analyzer</div>
    <div class="hero-subtitle">Weighted skill matching · Cosine similarity · ATS scoring for placement teams</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown('<div class="card-title">UPLOAD RESUMES</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    if uploaded_files:
        st.markdown(f"""
        <div class="card" style="text-align:center; padding: 1rem;">
            <div style="font-family: Space Mono, monospace; font-size: 1.8rem;
                        color: #38bdf8; font-weight: 700;">{len(uploaded_files)}</div>
            <div style="font-size: 0.7rem; color: #64748b; text-transform: uppercase;
                        letter-spacing: 0.05em;">Resumes Loaded</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)


# RESULTS
if uploaded_files:
    for idx, file in enumerate(uploaded_files):

        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 0.75rem; margin: 2rem 0 1rem;">
            <div style="background: #1e3a5f; border-radius: 8px; padding: 0.5rem 0.8rem;
                        font-family: Space Mono, monospace; font-size: 0.7rem; color: #38bdf8;">
                #{idx+1:02d}
            </div>
            <div style="font-size: 1rem; font-weight: 600; color: #e2e8f0;">
                {file.name}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Preprocessing
        raw_text = extract_text(file)
        cleaned_text = clean_text(raw_text)
        no_stop_text = remove_stopwords(cleaned_text)
        lemmatized_text = lemmatize_text(no_stop_text)
        word_count = len(lemmatized_text.split())

        # Cosine similarity
        has_jd = raw_jd is not None and raw_jd.strip() != ""
        cosine_score = 0

        if has_jd:
            cleaned_jd = clean_text(raw_jd)
            no_stop_jd = remove_stopwords(cleaned_jd)
            lemmatized_jd = lemmatize_text(no_stop_jd)
            cosine_score = get_cosine_similarity(lemmatized_text, lemmatized_jd)

        # Skill matching
        required_skills = {}
        if company and role:
            required_skills = get_required_skills(company, role)

        if required_skills:
            found, missing, match_percent = weighted_match(lemmatized_text, required_skills)
            final_score = get_final_score(match_percent, cosine_score, has_jd)
            score_class = get_score_class(final_score)

            m1, m2, m3, m4, m5 = st.columns(5)

            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value {score_class}">{final_score}%</div>
                    <div class="metric-label">Final ATS Score</div>
                </div>
                """, unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#94a3b8">{match_percent}%</div>
                    <div class="metric-label">Keyword Match</div>
                </div>
                """, unsafe_allow_html=True)

            with m3:
                cosine_display = f"{cosine_score}%" if has_jd else "N/A"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#a78bfa">{cosine_display}</div>
                    <div class="metric-label">Cosine Similarity</div>
                </div>
                """, unsafe_allow_html=True)

            with m4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#4ade80">{len(found)}</div>
                    <div class="metric-label">Skills Matched</div>
                </div>
                """, unsafe_allow_html=True)

            with m5:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color:#f87171">{len(missing)}</div>
                    <div class="metric-label">Skills Missing</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-bottom: 0.4rem; font-size: 0.75rem; color: #64748b;
                        font-family: Space Mono, monospace; text-transform: uppercase;
                        letter-spacing: 0.08em;">
                Match Progress
            </div>
            """, unsafe_allow_html=True)
            st.progress(final_score / 100)

            if not has_jd:
                st.markdown("""
                <div style="font-size: 0.75rem; color: #475569; margin-top: 0.5rem;
                            font-family: Space Mono, monospace;">
                    Switch to Paste JD mode to enable cosine similarity scoring.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["Skill Analysis", "Gap Chart", "Resume Preview"])

            with tab1:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="card-title">MATCHED SKILLS</div>', unsafe_allow_html=True)
                    if found:
                        tags = "".join([f'<span class="skill-tag skill-found">{s}</span>' for s in found])
                        st.markdown(f'<div style="line-height: 2.2">{tags}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="color: #475569; font-size: 0.85rem;">No skills matched.</div>', unsafe_allow_html=True)

                with c2:
                    st.markdown('<div class="card-title">MISSING SKILLS</div>', unsafe_allow_html=True)
                    if missing:
                        tags = "".join([f'<span class="skill-tag skill-missing">{s}</span>' for s in missing])
                        st.markdown(f'<div style="line-height: 2.2">{tags}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="color: #475569; font-size: 0.85rem;">All skills matched.</div>', unsafe_allow_html=True)

            with tab2:
                fig = plot_skill_gap(required_skills, found)
                st.pyplot(fig, use_container_width=True)

            with tab3:
                st.markdown('<div class="card-title">EXTRACTED TEXT PREVIEW</div>', unsafe_allow_html=True)
                st.text_area("", lemmatized_text[:800] + "...", height=200, label_visibility="collapsed")

        elif has_jd:
            st.markdown(f"""
            <div class="metric-card" style="max-width: 200px;">
                <div class="metric-value" style="color:#a78bfa">{cosine_score}%</div>
                <div class="metric-label">Cosine Similarity</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            st.progress(cosine_score / 100)

        else:
            st.warning("No skill data found for this company/role combination.")

        if idx < len(uploaded_files) - 1:
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <div style="font-family: Space Mono, monospace; font-size: 1rem;
                    color: #475569; margin-bottom: 0.5rem;">
            No resumes uploaded yet
        </div>
        <div style="font-size: 0.85rem; color: #334155;">
            Upload PDF resumes and select a company/role from the sidebar to begin analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)