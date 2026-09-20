import os
import re
import json
import tempfile
import textwrap
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.set_page_config(
    page_title="ResumeAI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    textwrap.dedent(
        """
        <style>

        /* Main background */
        .stApp {
            background: #f7f9fc;
        }

        /* Hide Streamlit branding */
        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }

        header {
            visibility: hidden;
        }

        /* Main content */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #0f172a;
        }

        section[data-testid="stSidebar"] * {
            color: white;
        }

        /* Logo */
        .logo {
            font-size: 28px;
            font-weight: 800;
            color: white;
            margin-bottom: 5px;
        }

        .logo-subtitle {
            font-size: 13px;
            color: #cbd5e1;
            margin-bottom: 25px;
        }

        /* Hero */
        .hero {
            background: linear-gradient(
                135deg,
                #0f172a 0%,
                #1e3a8a 100%
            );
            padding: 35px;
            border-radius: 18px;
            color: white;
            margin-bottom: 25px;
        }

        .hero h1 {
            font-size: 38px;
            margin: 0 0 10px 0;
            font-weight: 800;
        }

        .hero p {
            font-size: 16px;
            color: #dbeafe;
            margin: 0;
        }

        /* Cards */
        .card {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 15px;
            padding: 22px;
            margin-bottom: 18px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        .card-title {
            font-size: 19px;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 8px;
        }

        .card-text {
            color: #64748b;
            font-size: 14px;
        }

        /* Metric cards */
        .metric {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        }

        .metric-value {
            font-size: 30px;
            font-weight: 800;
            color: #1d4ed8;
        }

        .metric-label {
            color: #64748b;
            font-size: 13px;
            margin-top: 5px;
        }

        /* Section title */
        .section-title {
            font-size: 25px;
            font-weight: 800;
            color: #0f172a;
            margin: 10px 0 18px 0;
        }

        /* Candidate */
        .candidate-name {
            font-size: 21px;
            font-weight: 700;
            color: #0f172a;
        }

        .candidate-score {
            font-size: 28px;
            font-weight: 800;
            color: #2563eb;
        }

        /* Info box */
        .info-box {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            border-radius: 12px;
            padding: 15px;
            color: #1e40af;
            margin: 12px 0;
        }

        /* Success */
        .success-box {
            background: #f0fdf4;
            border: 1px solid #bbf7d0;
            border-radius: 12px;
            padding: 15px;
            color: #166534;
        }

        /* Warning */
        .warning-box {
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-radius: 12px;
            padding: 15px;
            color: #92400e;
        }

        /* Chat */
        .chat-user {
            background: #dbeafe;
            padding: 13px 16px;
            border-radius: 15px;
            margin: 8px 0;
            color: #1e3a8a;
        }

        .chat-ai {
            background: white;
            border: 1px solid #e2e8f0;
            padding: 15px 16px;
            border-radius: 15px;
            margin: 8px 0;
            color: #334155;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            font-weight: 600;
        }

        /* File uploader */
        section[data-testid="stFileUploader"] {
            background: white;
            border-radius: 12px;
        }

        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "resume_texts" not in st.session_state:
    st.session_state.resume_texts = {}

if "candidate_names" not in st.session_state:
    st.session_state.candidate_names = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0


# ============================================================
# LAZY LOADING FUNCTIONS
# ============================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    if not GOOGLE_API_KEY:
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )


# ============================================================
# PDF PROCESSING
# ============================================================

def extract_text_from_pdf(uploaded_file):
    """
    Extract text from a PDF using PyPDFLoader.
    """

    # Correct import
    from langchain_community.document_loaders import PyPDFLoader

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        pages = loader.load()

        return pages

    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass


def process_resumes(uploaded_files):
    """
    Process uploaded PDF resumes and create FAISS vector database.
    """

    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    all_documents = []
    resume_texts = {}
    candidate_names = []

    for uploaded_file in uploaded_files:

        try:

            pages = extract_text_from_pdf(uploaded_file)

            if not pages:
                continue

            # Store full text
            full_text = "\n".join(
                page.page_content
                for page in pages
            )

            resume_texts[uploaded_file.name] = full_text

            # Candidate name
            candidate_name = extract_candidate_name(
                full_text,
                uploaded_file.name
            )

            candidate_names.append(candidate_name)

            # Add metadata
            for page in pages:
                page.metadata["source"] = uploaded_file.name
                page.metadata["candidate"] = candidate_name

            all_documents.extend(pages)

        except Exception as e:
            st.error(
                f"Could not process {uploaded_file.name}: {e}"
            )

    if not all_documents:
        return False

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(all_documents)

    # Create embeddings
    embeddings = get_embeddings()

    # Create FAISS database
    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    st.session_state.documents = all_documents
    st.session_state.vectorstore = vectorstore
    st.session_state.resume_texts = resume_texts
    st.session_state.candidate_names = candidate_names
    st.session_state.processed_files = [
        file.name for file in uploaded_files
    ]
    st.session_state.chunks_count = len(chunks)

    return True


# ============================================================
# NAME EXTRACTION
# ============================================================

def extract_candidate_name(text, filename):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Try first few lines
    for line in lines[:10]:

        clean_line = re.sub(
            r"[^A-Za-z .'-]",
            "",
            line
        ).strip()

        words = clean_line.split()

        if 2 <= len(words) <= 5:
            if all(
                word.replace("-", "").replace("'", "").isalpha()
                for word in words
            ):
                return clean_line

    # Fallback filename
    name = Path(filename).stem

    name = re.sub(
        r"[_\-]+",
        " ",
        name
    )

    return name.title()


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_skills(text):

    common_skills = [
        "python",
        "java",
        "c",
        "c++",
        "c#",
        "javascript",
        "typescript",
        "html",
        "css",
        "react",
        "angular",
        "node.js",
        "node",
        "express",
        "spring",
        "spring boot",
        "sql",
        "mysql",
        "mongodb",
        "postgresql",
        "oracle",
        "git",
        "github",
        "docker",
        "kubernetes",
        "aws",
        "azure",
        "gcp",
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "ai",
        "generative ai",
        "genai",
        "llm",
        "langchain",
        "rag",
        "faiss",
        "streamlit",
        "pandas",
        "numpy",
        "tensorflow",
        "pytorch",
        "power bi",
        "excel",
        "figma",
        "ui/ux",
        "data analytics",
        "data analysis",
        "communication",
        "leadership",
    ]

    text_lower = text.lower()

    found = []

    for skill in common_skills:

        if skill.lower() in text_lower:
            found.append(skill)

    return sorted(set(found))


# ============================================================
# JOB MATCHING
# ============================================================

def calculate_match(resume_text, job_description):

    resume_lower = resume_text.lower()
    job_lower = job_description.lower()

    skills = extract_skills(job_description)

    if not skills:

        # General keyword matching
        job_words = set(
            re.findall(
                r"\b[a-zA-Z][a-zA-Z+#.-]{2,}\b",
                job_lower
            )
        )

        resume_words = set(
            re.findall(
                r"\b[a-zA-Z][a-zA-Z+#.-]{2,}\b",
                resume_lower
            )
        )

        matched = job_words.intersection(resume_words)

        score = min(
            100,
            int(
                len(matched) /
                max(len(job_words), 1)
                * 100
            )
        )

        return {
            "score": score,
            "matched": list(matched),
            "missing": [],
        }

    matched = []
    missing = []

    for skill in skills:

        if skill.lower() in resume_lower:
            matched.append(skill)
        else:
            missing.append(skill)

    score = int(
        len(matched) /
        max(len(skills), 1)
        * 100
    )

    return {
        "score": score,
        "matched": matched,
        "missing": missing,
    }


# ============================================================
# GEMINI ANSWER
# ============================================================

def ask_gemini(prompt):

    llm = get_llm()

    if llm is None:

        return (
            "Google API key is not configured. "
            "Please add GOOGLE_API_KEY in Render Environment Variables."
        )

    try:

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        return f"AI error: {str(e)}"


# ============================================================
# RAG QUESTION
# ============================================================

def ask_resume_question(question):

    if st.session_state.vectorstore is None:

        return (
            "Please upload and process at least one resume first."
        )

    try:

        from langchain_classic.chains import RetrievalQA

        llm = get_llm()

        if llm is None:

            return (
                "Google API key is not configured."
            )

        retriever = (
            st.session_state.vectorstore
            .as_retriever(
                search_kwargs={"k": 3}
            )
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
        )

        result = qa_chain.invoke(
            {"query": question}
        )

        return result["result"]

    except Exception as e:

        return f"Unable to answer the question: {str(e)}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="logo">ResumeAI</div>
        <div class="logo-subtitle">
            Intelligent Resume Analysis Platform
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "Dashboard",
            "Upload Resumes",
            "Job Match",
            "Compare Candidates",
            "AI Resume Chat",
        ],
        label_visibility="visible",
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="font-size:13px;color:#cbd5e1;">
        <b>Technology</b><br><br>
        • Python<br>
        • Streamlit<br>
        • LangChain<br>
        • Gemini AI<br>
        • FAISS<br>
        • HuggingFace
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.session_state.processed_files:

        st.success(
            f"{len(st.session_state.processed_files)} "
            f"resume(s) loaded"
        )

    else:

        st.info(
            "Upload resumes to begin."
        )


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

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

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-value">
                    {len(st.session_state.processed_files)}
                </div>
                <div class="metric-label">
                    Resumes
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-value">
                    {len(st.session_state.candidate_names)}
                </div>
                <div class="metric-label">
                    Candidates
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-value">
                    {st.session_state.chunks_count}
                </div>
                <div class="metric-label">
                    Text Chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col4:
        st.markdown(
            """
            <div class="metric">
                <div class="metric-value">
                    AI
                </div>
                <div class="metric-label">
                    Gemini Powered
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # Introduction
    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    📄 Resume Analysis
                </div>
                <div class="card-text">
                    Upload multiple PDF resumes and convert
                    them into searchable knowledge using
                    embeddings and FAISS vector search.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    🎯 Job Matching
                </div>
                <div class="card-text">
                    Compare resumes against a job description
                    and identify matched and missing skills.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    🤖 AI Resume Chat
                </div>
                <div class="card-text">
                    Ask natural-language questions about
                    uploaded resumes using Retrieval-Augmented
                    Generation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    📊 Candidate Comparison
                </div>
                <div class="card-text">
                    Compare candidate profiles, skills and
                    job-match scores in one workspace.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if not GOOGLE_API_KEY:

        st.markdown(
            """
            <div class="warning-box">
                ⚠️ GOOGLE_API_KEY is not configured.
                Add it in Render → Environment Variables
                before using Gemini AI features.
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# UPLOAD RESUMES
# ============================================================

elif page == "Upload Resumes":

    st.markdown(
        '<div class="section-title">Upload Resumes</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">
                📄 Add Candidate Resumes
            </div>
            <div class="card-text">
                Upload one or multiple PDF resumes.
                ResumeAI will extract the content, split it
                into chunks and create a searchable FAISS index.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose PDF resumes",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF resumes."
    )

    if uploaded_files:

        st.write(
            f"**{len(uploaded_files)} file(s) selected**"
        )

        for file in uploaded_files:

            st.write(
                f"📄 {file.name}"
            )

        if st.button(
            "Process Resumes",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Processing resumes and creating AI search index..."
            ):

                success = process_resumes(
                    uploaded_files
                )

            if success:

                st.success(
                    "Resumes processed successfully!"
                )

                st.rerun()

    if st.session_state.processed_files:

        st.markdown(
            '<div class="section-title">Loaded Resumes</div>',
            unsafe_allow_html=True
        )

        for filename in st.session_state.processed_files:

            candidate = st.session_state.resume_texts.get(
                filename,
                ""
            )

            name = extract_candidate_name(
                candidate,
                filename
            )

            skills = extract_skills(candidate)

            with st.expander(
                f"📄 {name} — {filename}"
            ):

                st.write(
                    f"**Detected skills:** "
                    f"{', '.join(skills) if skills else 'None detected'}"
                )

                st.write(
                    f"**Characters:** {len(candidate):,}"
                )


# ============================================================
# JOB MATCH
# ============================================================

elif page == "Job Match":

    st.markdown(
        '<div class="section-title">Job Match</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.resume_texts:

        st.info(
            "Upload resumes first from the Upload Resumes page."
        )

    else:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    🎯 Job Description
                </div>
                <div class="card-text">
                    Paste the job description below to compare
                    candidate resumes with the required skills.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        job_description = st.text_area(
            "Paste Job Description",
            value=st.session_state.job_description,
            height=250,
            placeholder=(
                "Example: We are looking for a Python developer "
                "with SQL, Git, AWS and machine learning experience..."
            )
        )

        st.session_state.job_description = job_description

        if st.button(
            "Analyze Job Match",
            type="primary",
            use_container_width=True
        ):

            if not job_description.strip():

                st.warning(
                    "Please enter a job description."
                )

            else:

                results = []

                for filename, resume_text in (
                    st.session_state.resume_texts.items()
                ):

                    result = calculate_match(
                        resume_text,
                        job_description
                    )

                    candidate_name = extract_candidate_name(
                        resume_text,
                        filename
                    )

                    result["name"] = candidate_name
                    result["filename"] = filename
                    result["resume_text"] = resume_text

                    results.append(result)

                results.sort(
                    key=lambda x: x["score"],
                    reverse=True
                )

                st.session_state.match_results = results

        if "match_results" in st.session_state:

            results = st.session_state.match_results

            st.markdown(
                '<div class="section-title">Match Results</div>',
                unsafe_allow_html=True
            )

            for index, result in enumerate(results):

                st.markdown(
                    f"""
                    <div class="card">
                        <div class="candidate-name">
                            #{index + 1} {result["name"]}
                        </div>

                        <div class="candidate-score">
                            {result["score"]}%
                        </div>

                        <p>
                            <b>Matched skills:</b>
                            {", ".join(result["matched"])
                            if result["matched"]
                            else "None detected"}
                        </p>

                        <p>
                            <b>Missing skills:</b>
                            {", ".join(result["missing"])
                            if result["missing"]
                            else "None"}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# ============================================================
# COMPARE CANDIDATES
# ============================================================

elif page == "Compare Candidates":

    st.markdown(
        '<div class="section-title">Compare Candidates</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.resume_texts:

        st.info(
            "Upload resumes first."
        )

    else:

        filenames = list(
            st.session_state.resume_texts.keys()
        )

        selected = st.multiselect(
            "Select candidates to compare",
            filenames,
            default=filenames[:2]
        )

        if selected:

            comparison_data = []

            for filename in selected:

                text = st.session_state.resume_texts[
                    filename
                ]

                name = extract_candidate_name(
                    text,
                    filename
                )

                skills = extract_skills(text)

                comparison_data.append(
                    {
                        "Candidate": name,
                        "Resume": filename,
                        "Skills": len(skills),
                        "Skill List": ", ".join(skills),
                    }
                )

            st.dataframe(
                comparison_data,
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                '<div class="section-title">Candidate Profiles</div>',
                unsafe_allow_html=True
            )

            columns = st.columns(
                min(len(selected), 3)
            )

            for index, filename in enumerate(selected):

                text = st.session_state.resume_texts[
                    filename
                ]

                name = extract_candidate_name(
                    text,
                    filename
                )

                skills = extract_skills(text)

                with columns[index % len(columns)]:

                    st.markdown(
                        f"""
                        <div class="card">
                            <div class="candidate-name">
                                {name}
                            </div>

                            <p>
                                <b>Resume:</b><br>
                                {filename}
                            </p>

                            <p>
                                <b>Skills:</b><br>
                                {", ".join(skills)
                                if skills
                                else "No common skills detected"}
                            </p>

                            <p>
                                <b>Skill Count:</b>
                                {len(skills)}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================
# AI RESUME CHAT
# ============================================================

elif page == "AI Resume Chat":

    st.markdown(
        '<div class="section-title">AI Resume Chat</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.vectorstore:

        st.markdown(
            """
            <div class="info-box">
                📄 Upload and process resumes first.
                After that you can ask questions about
                the resume content.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="card">
                <div class="card-title">
                    🤖 Ask questions about your resumes
                </div>
                <div class="card-text">
                    ResumeAI uses Retrieval-Augmented Generation
                    to retrieve relevant resume content before
                    generating an answer.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Previous messages
        for message in st.session_state.chat_history:

            if message["role"] == "user":

                st.markdown(
                    f"""
                    <div class="chat-user">
                        <b>You</b><br>
                        {message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="chat-ai">
                        <b>ResumeAI</b><br>
                        {message["content"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        question = st.chat_input(
            "Ask something about the uploaded resumes..."
        )

        if question:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.spinner(
                "Searching resumes and generating answer..."
            ):

                answer = ask_resume_question(
                    question
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

            st.rerun()

        if st.button(
            "Clear Chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        font-size:12px;
        padding:30px 0 10px 0;
    ">
        ResumeAI • RAG-powered Resume Intelligence
    </div>
    """,
    unsafe_allow_html=True
)
