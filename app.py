import os
import re
import json
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# =========================================================
# CONFIG
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="ResumeAI | Intelligent Recruitment",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# PROFESSIONAL UI
# =========================================================

st.markdown(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f6f8fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

[data-testid="stSidebar"] {
    background: #101828;
}

[data-testid="stSidebar"] * {
    color: #e6eaf0;
}

.brand {
    font-size: 26px;
    font-weight: 800;
    margin-bottom: 5px;
}

.brand-sub {
    color: #98a2b3 !important;
    font-size: 12px;
    margin-bottom: 30px;
}

.hero {
    background: linear-gradient(135deg, #101828 0%, #1d2939 100%);
    padding: 34px;
    border-radius: 24px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 38px;
    margin: 0 0 8px 0;
    font-weight: 800;
}

.hero p {
    color: #cbd5e1;
    margin: 0;
    font-size: 15px;
}

.card {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 18px;
    box-shadow: 0 3px 14px rgba(16,24,40,0.04);
}

.card-title {
    font-size: 18px;
    font-weight: 700;
    color: #101828;
    margin-bottom: 4px;
}

.card-sub {
    color: #667085;
    font-size: 13px;
    margin-bottom: 16px;
}

.metric {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 16px;
    padding: 20px;
}

.metric-label {
    color: #667085;
    font-size: 12px;
    font-weight: 600;
}

.metric-value {
    color: #101828;
    font-size: 28px;
    font-weight: 800;
    margin-top: 4px;
}

.candidate {
    background: white;
    border: 1px solid #e4e7ec;
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 15px;
}

.candidate-name {
    font-size: 19px;
    font-weight: 800;
    color: #101828;
}

.score {
    font-size: 30px;
    font-weight: 800;
}

.skill {
    display: inline-block;
    background: #eef4ff;
    color: #344054;
    border: 1px solid #dbe7ff;
    border-radius: 20px;
    padding: 5px 10px;
    margin: 3px;
    font-size: 12px;
}

.small {
    color: #667085;
    font-size: 12px;
}

.info-box {
    background: #f8fafc;
    border-left: 4px solid #667085;
    border-radius: 10px;
    padding: 14px;
    margin: 10px 0;
    color: #344054;
    font-size: 13px;
}

div.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #98a2b3;
    font-size: 12px;
    padding: 30px 0 5px 0;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SESSION STATE
# =========================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "candidate_data" not in st.session_state:
    st.session_state.candidate_data = {}

if "job_description" not in st.session_state:
    st.session_state.job_description = ""


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_name(filename):
    name = Path(filename).stem
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()


def get_llm():

    if not GOOGLE_API_KEY:
        st.error(
            "GOOGLE_API_KEY is missing. "
            "Please add GOOGLE_API_KEY in Render → Environment."
        )
        st.stop()

    # Import only when needed
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )


@st.cache_resource(show_spinner=False)
def get_embeddings():

    # Heavy library is loaded only when embeddings are actually needed
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


def extract_candidate_profile(llm, candidate_name, text):

    prompt = f"""
You are an HR resume analysis assistant.

Analyze this resume for candidate: {candidate_name}

Return ONLY valid JSON with these keys:

{{
  "name": "candidate name if found, otherwise {candidate_name}",
  "skills": ["skill1", "skill2"],
  "education": ["education item"],
  "experience": ["experience item"],
  "strengths": ["strength1", "strength2", "strength3"],
  "summary": "short professional summary"
}}

Do not invent information.
Use only information present in the resume.

RESUME:
{text[:12000]}
"""

    try:

        response = llm.invoke(prompt)
        raw = response.content

        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            return {
                "name": candidate_name,
                "skills": [],
                "education": [],
                "experience": [],
                "strengths": [],
                "summary": "Profile could not be structured."
            }

        return json.loads(match.group())

    except Exception as e:

        return {
            "name": candidate_name,
            "skills": [],
            "education": [],
            "experience": [],
            "strengths": [],
            "summary": "Profile analysis unavailable."
        }


def analyze_match(llm, candidate_name, resume_text, job_description):

    prompt = f"""
You are an explainable recruitment assistant.

Compare the candidate resume with the job description.

Return ONLY valid JSON:

{{
  "score": 0,
  "matched_skills": ["skill"],
  "missing_skills": ["skill"],
  "strengths": ["reason"],
  "gaps": ["reason"],
  "recommendation": "Strong Match / Good Match / Partial Match / Low Match",
  "explanation": "short explanation"
}}

Rules:

- score must be an integer from 0 to 100.
- Base the score ONLY on the provided resume and job description.
- Do not invent experience, education, certifications, or skills.
- This is an AI-assisted recommendation, not a hiring decision.

CANDIDATE:
{candidate_name}

RESUME:
{resume_text[:12000]}

JOB DESCRIPTION:
{job_description[:8000]}
"""

    try:

        response = llm.invoke(prompt)
        raw = response.content

        match = re.search(r"\{.*\}", raw, re.DOTALL)

        if not match:
            return None

        data = json.loads(match.group())

        data["score"] = max(
            0,
            min(
                100,
                int(data.get("score", 0))
            )
        )

        return data

    except Exception:
        return None


def candidate_text(candidate_name):

    parts = []

    for doc in st.session_state.documents:

        source = str(
            doc.metadata.get("source", "")
        )

        if Path(source).stem.lower() == candidate_name.lower():

            parts.append(
                doc.page_content
            )

    if not parts:

        for doc in st.session_state.documents:

            source_name = clean_name(
                Path(
                    str(
                        doc.metadata.get(
                            "source",
                            ""
                        )
                    )
                ).name
            )

            if source_name.lower() == candidate_name.lower():

                parts.append(
                    doc.page_content
                )

    return "\n".join(parts)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="brand">✦ ResumeAI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-sub">'
        'AI-powered recruitment intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "WORKSPACE",
        [
            "Dashboard",
            "Upload Resumes",
            "Job Match",
            "Compare Candidates",
            "AI Resume Chat",
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("### AI Workflow")

    st.markdown("① Upload resumes")
    st.markdown("② Add job description")
    st.markdown("③ Analyze candidates")
    st.markdown("④ Compare & rank")
    st.markdown("⑤ Ask your resumes")

    st.markdown("---")

    st.caption("ResumeAI • RAG + Gemini")


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
<div class="hero">

    <h1>Resume Intelligence</h1>

    <p>
    Search, rank, compare and understand candidates
    using Retrieval-Augmented Generation.
    </p>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            f"""
            <div class="metric">

                <div class="metric-label">
                    RESUMES
                </div>

                <div class="metric-value">
                    {len(st.session_state.candidate_data)}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric">

                <div class="metric-label">
                    PDF PAGES
                </div>

                <div class="metric-value">
                    {len(st.session_state.documents)}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric">

                <div class="metric-label">
                    TEXT CHUNKS
                </div>

                <div class="metric-value">
                    {len(st.session_state.chunks)}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:

        scores = [
            x.get("match", {}).get("score")
            for x in st.session_state.candidate_data.values()
            if x.get("match")
        ]

        avg = (
            round(sum(scores) / len(scores))
            if scores
            else 0
        )

        st.markdown(
            f"""
            <div class="metric">

                <div class="metric-label">
                    AVG MATCH
                </div>

                <div class="metric-value">
                    {avg}%
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Recruitment Command Center
            </div>

            <div class="card-sub">
                Your AI workspace for resume screening
                and candidate intelligence.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.candidate_data:

        st.info(
            "Start by opening **Upload Resumes** "
            "and uploading one or more PDF resumes."
        )

    else:

        ranked = sorted(
            st.session_state.candidate_data.items(),
            key=lambda x: x[1].get(
                "match",
                {}
            ).get(
                "score",
                0
            ),
            reverse=True
        )

        st.subheader("Top Candidates")

        for rank, (name, data) in enumerate(
            ranked[:5],
            1
        ):

            score = data.get(
                "match",
                {}
            ).get(
                "score",
                0
            )

            c1, c2 = st.columns([5, 1])

            with c1:

                st.markdown(
                    f"""
                    <div class="candidate">

                        <div class="candidate-name">
                            #{rank} &nbsp; {name}
                        </div>

                        <div class="small">
                            {data.get("profile", {}).get(
                                "summary",
                                ""
                            )}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with c2:

                st.markdown(
                    f"""
                    <div class="candidate">

                        <div class="score">
                            {score}%
                        </div>

                        <div class="small">
                            AI Match
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


# =========================================================
# UPLOAD RESUMES
# =========================================================

elif page == "Upload Resumes":

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Upload Candidate Resumes
            </div>

            <div class="card-sub">
                Upload multiple PDF resumes.
                ResumeAI will create embeddings
                and candidate profiles.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose resume PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        if st.button(
            "⚡ Process Resumes",
            use_container_width=True
        ):

            documents = []
            candidate_data = {}

            with st.spinner(
                "Reading resumes and building AI knowledge base..."
            ):

                # -----------------------------------------
                # Load PDF
                # -----------------------------------------

                from langchain_community.document_loaders import (
                    PyPDFLoader
                )

                for uploaded in uploaded_files:

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as tmp:

                        tmp.write(
                            uploaded.getbuffer()
                        )

                        temp_path = tmp.name

                    try:

                        loader = PyPDFLoader(
                            temp_path
                        )

                        docs = loader.load()

                        for doc in docs:

                            doc.metadata["source"] = (
                                uploaded.name
                            )

                        documents.extend(docs)

                    finally:

                        try:
                            os.remove(temp_path)
                        except OSError:
                            pass

                # -----------------------------------------
                # Split documents
                # -----------------------------------------

                from langchain_text_splitters import (
                    RecursiveCharacterTextSplitter
                )

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(
                    documents
                )

                if not chunks:

                    st.error(
                        "No readable text was found "
                        "inside the uploaded PDF."
                    )

                    st.stop()

                # -----------------------------------------
                # Embeddings
                # -----------------------------------------

                st.info(
                    "Loading the AI embedding model. "
                    "The first time may take a little longer."
                )

                embeddings = get_embeddings()

                # -----------------------------------------
                # FAISS
                # -----------------------------------------

                from langchain_community.vectorstores import (
                    FAISS
                )

                vector_db = FAISS.from_documents(
                    chunks,
                    embeddings
                )

                retriever = vector_db.as_retriever(
                    search_kwargs={
                        "k": 4
                    }
                )

                # -----------------------------------------
                # Gemini
                # -----------------------------------------

                llm = get_llm()

                from langchain_classic.chains import (
                    RetrievalQA
                )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=retriever
                )

                # -----------------------------------------
                # Candidate profiles
                # -----------------------------------------

                grouped = {}

                for doc in documents:

                    source = Path(
                        str(
                            doc.metadata.get(
                                "source",
                                "resume.pdf"
                            )
                        )
                    ).name

                    grouped.setdefault(
                        source,
                        []
                    ).append(
                        doc.page_content
                    )

                for source, pages in grouped.items():

                    name = clean_name(
                        source
                    )

                    full_text = "\n".join(
                        pages
                    )

                    profile = extract_candidate_profile(
                        llm,
                        name,
                        full_text
                    )

                    candidate_data[name] = {

                        "profile": profile,

                        "resume_text": full_text,

                        "match": None,
                    }

                # -----------------------------------------
                # Save session data
                # -----------------------------------------

                st.session_state.documents = documents

                st.session_state.chunks = chunks

                st.session_state.vector_db = vector_db

                st.session_state.qa_chain = qa_chain

                st.session_state.candidate_data = (
                    candidate_data
                )

            st.success(
                f"Successfully processed "
                f"{len(candidate_data)} "
                f"candidate resume(s)."
            )

            st.rerun()


# =========================================================
# JOB MATCH
# =========================================================

elif page == "Job Match":

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                🎯 AI Job Matching
            </div>

            <div class="card-sub">
                Paste a job description and let AI
                analyze every uploaded candidate.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if not st.session_state.candidate_data:

        st.warning(
            "Upload resumes first."
        )

    else:

        jd = st.text_area(
            "Job Description",
            value=st.session_state.job_description,
            height=230,
            placeholder=(
                "Paste the complete job description here..."
            )
        )

        st.session_state.job_description = jd

        if st.button(
            "🚀 Analyze Candidate Matches",
            use_container_width=True
        ):

            if not jd.strip():

                st.warning(
                    "Please enter a job description."
                )

            else:

                llm = get_llm()

                progress = st.progress(0)

                total = len(
                    st.session_state.candidate_data
                )

                for i, (
                    name,
                    data
                ) in enumerate(
                    st.session_state.candidate_data.items()
                ):

                    result = analyze_match(
                        llm,
                        name,
                        data["resume_text"],
                        jd
                    )

                    st.session_state.candidate_data[
                        name
                    ]["match"] = result

                    progress.progress(
                        (i + 1) / total
                    )

                st.success(
                    "Candidate matching completed."
                )

                st.rerun()

        matches = [
            (name, data)
            for name, data
            in st.session_state.candidate_data.items()
            if data.get("match")
        ]

        if matches:

            ranked = sorted(
                matches,
                key=lambda x: x[1][
                    "match"
                ]["score"],
                reverse=True
            )

            st.markdown(
                "### 🏆 Candidate Ranking"
            )

            for rank, (
                name,
                data
            ) in enumerate(
                ranked,
                1
            ):

                match = data["match"]

                score = match["score"]

                st.markdown(
                    f"""
                    <div class="candidate">

                        <div class="candidate-name">
                            #{rank} {name}
                        </div>

                        <div class="small">
                            {match.get(
                                "recommendation",
                                ""
                            )}
                        </div>

                        <br>

                        <div class="score">
                            {score}%
                        </div>

                        <div class="small">
                            AI-assisted job match score
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "**Matched Skills**"
                    )

                    for skill in match.get(
                        "matched_skills",
                        []
                    ):

                        st.markdown(
                            f"""
                            <span class="skill">
                                ✓ {skill}
                            </span>
                            """,
                            unsafe_allow_html=True
                        )

                with col2:

                    st.markdown(
                        "**Skill Gaps**"
                    )

                    for skill in match.get(
                        "missing_skills",
                        []
                    ):

                        st.markdown(
                            f"""
                            <span class="skill">
                                ＋ {skill}
                            </span>
                            """,
                            unsafe_allow_html=True
                        )

                with st.expander(
                    "Why this score?"
                ):

                    st.write(
                        match.get(
                            "explanation",
                            "No explanation available."
                        )
                    )

                    if match.get(
                        "strengths"
                    ):

                        st.markdown(
                            "**Strengths**"
                        )

                        for item in match[
                            "strengths"
                        ]:

                            st.write(
                                "•",
                                item
                            )

                    if match.get(
                        "gaps"
                    ):

                        st.markdown(
                            "**Gaps**"
                        )

                        for item in match[
                            "gaps"
                        ]:

                            st.write(
                                "•",
                                item
                            )


# =========================================================
# COMPARE CANDIDATES
# =========================================================

elif page == "Compare Candidates":

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                ⚖️ Candidate Comparison
            </div>

            <div class="card-sub">
                Compare two candidates side-by-side
                using their extracted profiles.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    names = list(
        st.session_state.candidate_data.keys()
    )

    if len(names) < 2:

        st.warning(
            "Upload at least two resumes "
            "to use candidate comparison."
        )

    else:

        col1, col2 = st.columns(2)

        with col1:

            candidate_a = st.selectbox(
                "Candidate A",
                names,
                index=0
            )

        with col2:

            candidate_b = st.selectbox(
                "Candidate B",
                names,
                index=1
            )

        if candidate_a == candidate_b:

            st.warning(
                "Please select two different candidates."
            )

        else:

            a = st.session_state.candidate_data[
                candidate_a
            ]

            b = st.session_state.candidate_data[
                candidate_b
            ]

            ca, cb = st.columns(2)

            for column, name, data in [
                (ca, candidate_a, a),
                (cb, candidate_b, b)
            ]:

                with column:

                    profile = data["profile"]

                    match = data.get(
                        "match"
                    ) or {}

                    st.markdown(
                        f"""
                        <div class="candidate">

                            <div class="candidate-name">
                                {name}
                            </div>

                            <p>
                                {profile.get(
                                    "summary",
                                    ""
                                )}
                            </p>

                            <hr>
                        """,
                        unsafe_allow_html=True
                    )

                    if match:

                        st.markdown(
                            f"""
                            <div class="score">
                                {match.get(
                                    "score",
                                    0
                                )}%
                            </div>

                            <div class="small">
                                Job Match
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        "**Skills**"
                    )

                    for skill in profile.get(
                        "skills",
                        []
                    ):

                        st.markdown(
                            f"""
                            <span class="skill">
                                {skill}
                            </span>
                            """,
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        "**Strengths**"
                    )

                    for item in profile.get(
                        "strengths",
                        []
                    ):

                        st.write(
                            "•",
                            item
                        )

                    st.markdown(
                        "</div>",
                        unsafe_allow_html=True
                    )


# =========================================================
# AI RESUME CHAT
# =========================================================

elif page == "AI Resume Chat":

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                💬 Ask Your Resumes
            </div>

            <div class="card-sub">
                Ask natural-language questions and retrieve
                information from the uploaded resumes.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.qa_chain is None:

        st.info(
            "Upload and process resumes first."
        )

    else:

        examples = [

            "Who has customer service experience?",

            "Which candidate has strong communication skills?",

            "Find candidates with leadership experience.",

            "Which resume mentions Python?",
        ]

        selected = st.selectbox(
            "Quick questions",
            [
                "Choose a question..."
            ] + examples
        )

        question = st.text_input(
            "Your question",
            value=(
                ""
                if selected == "Choose a question..."
                else selected
            ),
            placeholder=(
                "Ask anything about the uploaded resumes..."
            )
        )

        if st.button(
            "🔎 Search Resumes",
            use_container_width=True
        ):

            if question.strip():

                with st.spinner(
                    "Searching resumes..."
                ):

                    try:

                        result = (
                            st.session_state
                            .qa_chain
                            .invoke(
                                {
                                    "query": question
                                }
                            )
                        )

                        answer = result.get(
                            "result",
                            "No answer found."
                        )

                        st.markdown(
                            "### AI Response"
                        )

                        st.markdown(
                            f"""
                            <div class="card">
                                {answer}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    except Exception as e:

                        st.error(
                            "Unable to process the question."
                        )

                        st.caption(
                            str(e)
                        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        ResumeAI • RAG • FAISS • HuggingFace Embeddings • Gemini
    </div>
    """,
    unsafe_allow_html=True
)
