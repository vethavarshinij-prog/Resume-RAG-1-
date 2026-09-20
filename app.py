import os
import re
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
    page_title="ResumeAI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f7f9fc;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    /* Main */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Hero */
    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 35px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 38px;
        margin: 0;
        font-weight: 800;
    }

    .hero p {
        font-size: 16px;
        color: #dbeafe;
    }

    /* Cards */
    .card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 2px 8px rgba(15,23,42,0.04);
    }

    .card-title {
        font-size: 19px;
        font-weight: 700;
        color: #0f172a;
    }

    .card-text {
        color: #64748b;
        font-size: 14px;
        margin-top: 8px;
    }

    /* Metrics */
    .metric {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 22px;
        text-align: center;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #2563eb;
    }

    .metric-label {
        color: #64748b;
        margin-top: 5px;
    }

    /* Section */
    .section-title {
        font-size: 25px;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 20px;
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

    /* Chat */
    .chat-user {
        background: #dbeafe;
        padding: 14px;
        border-radius: 14px;
        margin: 8px 0;
    }

    .chat-ai {
        background: white;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 14px;
        margin: 8px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# SESSION STATE
# =========================================================

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

if "chunks_count" not in st.session_state:
    st.session_state.chunks_count = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "match_results" not in st.session_state:
    st.session_state.match_results = []


# =========================================================
# EMBEDDINGS
# =========================================================

@st.cache_resource(show_spinner=False)
def get_embeddings():

    from langchain_community.embeddings import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    return embeddings


# =========================================================
# GEMINI LLM
# =========================================================

@st.cache_resource(show_spinner=False)
def get_llm():

    if not GOOGLE_API_KEY:
        return None

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )


# =========================================================
# PDF READER
# =========================================================

def extract_pdf(uploaded_file):

    from langchain_community.document_loaders import PyPDFLoader

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:

            temp.write(uploaded_file.getbuffer())
            temp_path = temp.name

        loader = PyPDFLoader(temp_path)

        pages = loader.load()

        return pages

    finally:

        if temp_path:

            try:
                os.remove(temp_path)
            except Exception:
                pass


# =========================================================
# NAME EXTRACTION
# =========================================================

def extract_name(text, filename):

    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    for line in lines[:10]:

        cleaned = re.sub(
            r"[^A-Za-z .'-]",
            "",
            line
        ).strip()

        words = cleaned.split()

        if 2 <= len(words) <= 5:

            if all(
                word.replace("-", "").replace("'", "").isalpha()
                for word in words
            ):

                return cleaned

    return Path(filename).stem.replace(
        "_",
        " "
    ).title()


# =========================================================
# SKILLS
# =========================================================

def extract_skills(text):

    skills = [
        "Python",
        "Java",
        "C",
        "C++",
        "C#",
        "JavaScript",
        "HTML",
        "CSS",
        "React",
        "Angular",
        "Node.js",
        "SQL",
        "MySQL",
        "MongoDB",
        "Git",
        "GitHub",
        "Docker",
        "AWS",
        "Azure",
        "Machine Learning",
        "Deep Learning",
        "Artificial Intelligence",
        "Generative AI",
        "LLM",
        "LangChain",
        "RAG",
        "FAISS",
        "Streamlit",
        "Pandas",
        "NumPy",
        "TensorFlow",
        "PyTorch",
        "Power BI",
        "Excel",
        "Figma",
        "UI/UX",
        "Data Analytics"
    ]

    text_lower = text.lower()

    return [
        skill
        for skill in skills
        if skill.lower() in text_lower
    ]


# =========================================================
# PROCESS RESUMES
# =========================================================

def process_resumes(files):

    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter
    )

    from langchain_community.vectorstores import FAISS

    all_pages = []
    resume_texts = {}
    names = []

    total_files = len(files)

    # -----------------------------------------------------
    # STEP 1 - READ PDFs
    # -----------------------------------------------------

    st.write("### 📄 Reading resumes...")

    pdf_progress = st.progress(0)

    for index, file in enumerate(files):

        try:

            st.write(
                f"📖 Reading **{file.name}**..."
            )

            pages = extract_pdf(file)

            if not pages:
                st.warning(
                    f"⚠️ No readable pages found in {file.name}"
                )
                continue

            full_text = "\n".join(
                page.page_content
                for page in pages
            )

            if not full_text.strip():

                st.warning(
                    f"⚠️ No text found in {file.name}"
                )
                continue

            name = extract_name(
                full_text,
                file.name
            )

            for page in pages:

                page.metadata["source"] = file.name
                page.metadata["candidate"] = name

            all_pages.extend(pages)

            resume_texts[file.name] = full_text

            names.append(name)

            st.success(
                f"✅ {file.name} — {len(pages)} page(s)"
            )

        except Exception as e:

            st.error(
                f"❌ Error processing {file.name}: {e}"
            )

        pdf_progress.progress(
            (index + 1) / total_files
        )

    if not all_pages:

        st.error(
            "❌ No readable resume content was found."
        )

        return False

    # -----------------------------------------------------
    # STEP 2 - SPLIT TEXT
    # -----------------------------------------------------

    st.write("### 🧩 Preparing resume text...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )

    chunks = splitter.split_documents(
        all_pages
    )

    st.info(
        f"📄 Pages: **{len(all_pages)}**  |  "
        f"🧩 Text chunks: **{len(chunks)}**"
    )

    if not chunks:

        st.error(
            "❌ Could not create text chunks."
        )

        return False

    # -----------------------------------------------------
    # STEP 3 - LOAD EMBEDDING MODEL
    # -----------------------------------------------------

    st.write("### 🤖 Preparing AI search model...")

    embedding_status = st.empty()

    embedding_status.info(
        "⏳ Loading the embedding model..."
    )

    try:

        embeddings = get_embeddings()

        embedding_status.success(
            "✅ Embedding model ready."
        )

    except Exception as e:

        embedding_status.error(
            f"❌ Could not load embedding model: {e}"
        )

        return False

    # -----------------------------------------------------
    # STEP 4 - CREATE FAISS DATABASE
    # -----------------------------------------------------

    st.write("### 🔍 Creating searchable resume database...")

    faiss_status = st.empty()

    faiss_status.info(
        "⏳ Creating FAISS vector database..."
    )

    try:

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        faiss_status.success(
            "✅ Search database created."
        )

    except Exception as e:

        faiss_status.error(
            f"❌ FAISS creation failed: {e}"
        )

        return False

    # -----------------------------------------------------
    # STEP 5 - SAVE RESULTS
    # -----------------------------------------------------

    st.write("### 💾 Saving processed resume data...")

    st.session_state.documents = all_pages

    st.session_state.vectorstore = vectorstore

    st.session_state.resume_texts = resume_texts

    st.session_state.candidate_names = names

    st.session_state.processed_files = list(
        resume_texts.keys()
    )

    st.session_state.chunks_count = len(chunks)

    # Clear previous job matching results
    st.session_state.match_results = []

    st.success(
        f"🎉 Successfully processed "
        f"{len(resume_texts)} resume(s)!"
    )

    return True


# =========================================================
# JOB MATCH
# =========================================================

def calculate_match(resume, job):

    resume_lower = resume.lower()
    job_lower = job.lower()

    job_skills = extract_skills(job)

    matched = []
    missing = []

    for skill in job_skills:

        if skill.lower() in resume_lower:

            matched.append(skill)

        else:

            missing.append(skill)

    if job_skills:

        score = int(
            len(matched)
            / len(job_skills)
            * 100
        )

    else:

        job_words = set(
            re.findall(
                r"\b[a-zA-Z]{3,}\b",
                job_lower
            )
        )

        resume_words = set(
            re.findall(
                r"\b[a-zA-Z]{3,}\b",
                resume_lower
            )
        )

        common = job_words.intersection(
            resume_words
        )

        score = int(
            len(common)
            / max(len(job_words), 1)
            * 100
        )

        score = min(
            score,
            100
        )

    return score, matched, missing


# =========================================================
# AI RESUME CHAT
# =========================================================

def ask_resume(question):

    if not GOOGLE_API_KEY:

        return (
            "GOOGLE_API_KEY is not configured "
            "in the environment variables."
        )

    if not st.session_state.vectorstore:

        return (
            "Please upload and process "
            "a resume first."
        )

    try:

        llm = get_llm()

        if llm is None:

            return (
                "Gemini could not be initialized."
            )

        retriever = (
            st.session_state.vectorstore
            .as_retriever(
                search_kwargs={
                    "k": 4
                }
            )
        )

        documents = retriever.invoke(
            question
        )

        if not documents:

            return (
                "The requested information "
                "is not available in the resume."
            )

        context = "\n\n".join(
            doc.page_content
            for doc in documents
        )

        prompt = f"""
You are ResumeAI, an AI assistant that answers
questions about uploaded resumes.

Use ONLY the resume information provided below.

If the answer is not present in the resume,
say that the information is not available.

Do not invent information.

RESUME CONTENT:
{context}

QUESTION:
{question}

Give a clear and concise answer.
"""

        response = llm.invoke(
            prompt
        )

        return response.content

    except Exception as e:

        return f"AI error: {e}"


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:28px;
            font-weight:800;
            color:white;
        ">
            ResumeAI
        </div>

        <div style="
            font-size:13px;
            color:#cbd5e1;
            margin-bottom:20px;
        ">
            Intelligent Resume Analysis
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
            "AI Resume Chat"
        ]
    )

    st.markdown("---")

    if st.session_state.processed_files:

        st.success(
            f"{len(st.session_state.processed_files)} "
            "resume(s) loaded"
        )

    else:

        st.info(
            "Upload resumes to begin."
        )


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="hero">
            <h1>Resume Intelligence</h1>

            <p>
                Search, rank, compare and understand
                candidates using Retrieval-Augmented Generation.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

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

    with c2:

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

    with c3:

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

    with c4:

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

    st.write("")

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    📄 Resume Analysis
                </div>

                <div class="card-text">
                    Upload PDF resumes and convert them into
                    searchable knowledge using embeddings
                    and FAISS vector search.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:

        st.markdown(
            """
            <div class="card">

                <div class="card-title">
                    🤖 AI Resume Chat
                </div>

                <div class="card-text">
                    Ask natural-language questions about
                    uploaded resumes using RAG.
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    if not GOOGLE_API_KEY:

        st.warning(
            "⚠️ GOOGLE_API_KEY is not configured."
        )


# =========================================================
# UPLOAD RESUMES
# =========================================================

elif page == "Upload Resumes":

    st.markdown(
        '<div class="section-title">📄 Upload Resumes</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                Upload Candidate Resumes
            </div>

            <div class="card-text">
                Select one or more PDF resumes and click
                Process Resumes.
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose PDF resumes",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(
            f"Selected **{len(uploaded_files)}** file(s)"
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

            with st.status(
                "🚀 Processing resumes...",
                expanded=True
            ) as status:

                success = process_resumes(
                    uploaded_files
                )

                if success:

                    status.update(
                        label="✅ Resume processing completed!",
                        state="complete",
                        expanded=False
                    )

                else:

                    status.update(
                        label="❌ Resume processing failed.",
                        state="error",
                        expanded=True
                    )

            if success:

                st.balloons()

                st.success(
                    "🎉 Resumes are ready to search!"
                )

    # -----------------------------------------------------
    # LOADED RESUMES
    # -----------------------------------------------------

    if st.session_state.processed_files:

        st.markdown(
            '<div class="section-title">Loaded Resumes</div>',
            unsafe_allow_html=True
        )

        for filename in st.session_state.processed_files:

            text = st.session_state.resume_texts[
                filename
            ]

            name = extract_name(
                text,
                filename
            )

            skills = extract_skills(
                text
            )

            with st.expander(
                f"📄 {name}"
            ):

                st.write(
                    f"**File:** {filename}"
                )

                st.write(
                    f"**Skills:** "
                    f"{', '.join(skills) if skills else 'None detected'}"
                )


# =========================================================
# JOB MATCH
# =========================================================

elif page == "Job Match":

    st.markdown(
        '<div class="section-title">🎯 Job Match</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.resume_texts:

        st.info(
            "Please upload resumes first."
        )

    else:

        job_description = st.text_area(
            "Paste Job Description",
            height=250,
            placeholder=(
                "Example: Looking for a Python developer "
                "with SQL, AWS and Machine Learning skills..."
            )
        )

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

                with st.spinner(
                    "Analyzing resumes..."
                ):

                    results = []

                    for filename, resume in (
                        st.session_state.resume_texts.items()
                    ):

                        score, matched, missing = (
                            calculate_match(
                                resume,
                                job_description
                            )
                        )

                        name = extract_name(
                            resume,
                            filename
                        )

                        results.append(
                            {
                                "name": name,
                                "filename": filename,
                                "score": score,
                                "matched": matched,
                                "missing": missing
                            }
                        )

                    results.sort(
                        key=lambda x: x["score"],
                        reverse=True
                    )

                    st.session_state.match_results = results

        for result in st.session_state.match_results:

            st.markdown(
                f"""
                <div class="card">

                    <div class="candidate-name">
                        {result["name"]}
                    </div>

                    <div class="candidate-score">
                        {result["score"]}%
                    </div>

                    <p>
                        <b>Matched skills:</b>
                        {
                            ", ".join(result["matched"])
                            if result["matched"]
                            else "None"
                        }
                    </p>

                    <p>
                        <b>Missing skills:</b>
                        {
                            ", ".join(result["missing"])
                            if result["missing"]
                            else "None"
                        }
                    </p>

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# COMPARE CANDIDATES
# =========================================================

elif page == "Compare Candidates":

    st.markdown(
        '<div class="section-title">📊 Compare Candidates</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.resume_texts:

        st.info(
            "Please upload resumes first."
        )

    else:

        filenames = list(
            st.session_state.resume_texts.keys()
        )

        selected = st.multiselect(
            "Select candidates",
            filenames,
            default=filenames[:2]
        )

        if selected:

            data = []

            for filename in selected:

                text = st.session_state.resume_texts[
                    filename
                ]

                name = extract_name(
                    text,
                    filename
                )

                skills = extract_skills(
                    text
                )

                data.append(
                    {
                        "Candidate": name,
                        "Resume": filename,
                        "Skill Count": len(skills),
                        "Skills": ", ".join(skills)
                    }
                )

            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True
            )


# =========================================================
# AI RESUME CHAT
# =========================================================

elif page == "AI Resume Chat":

    st.markdown(
        '<div class="section-title">🤖 AI Resume Chat</div>',
        unsafe_allow_html=True
    )

    if not st.session_state.vectorstore:

        st.info(
            "Please upload and process a resume first."
        )

    else:

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
            "Ask something about the uploaded resume..."
        )

        if question:

            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question
                }
            )

            with st.spinner(
                "🔍 Searching resume and generating answer..."
            ):

                answer = ask_resume(
                    question
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            st.rerun()

        if st.button(
            "Clear Chat",
            use_container_width=True
        ):

            st.session_state.chat_history = []

            st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div style="
        text-align:center;
        color:#94a3b8;
        font-size:12px;
        padding:30px 0 10px;
    ">
        ResumeAI • RAG-powered Resume Intelligence
    </div>
    """,
    unsafe_allow_html=True
)
