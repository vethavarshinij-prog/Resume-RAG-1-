import os
import re
import tempfile
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
    initial_sidebar_state="expanded"
)

# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "documents": [],
    "vectorstore": None,
    "resume_texts": {},
    "candidate_names": {},
    "processed_files": [],
    "chunk_count": 0,
    "chat_history": [],
    "match_results": []
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):
    """Clean extracted PDF text."""
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_name(text, filename):
    """Try to identify candidate name from resume text."""

    lines = [
        line.strip()
        for line in text.split("\n")
        if line.strip()
    ]

    # Look for common name headings
    for line in lines[:15]:
        clean_line = re.sub(r"[^A-Za-z .'-]", "", line).strip()

        words = clean_line.split()

        if (
            2 <= len(words) <= 5
            and len(clean_line) >= 4
            and len(clean_line) <= 60
        ):
            lower = clean_line.lower()

            excluded = [
                "resume",
                "curriculum vitae",
                "curriculum",
                "objective",
                "profile",
                "summary",
                "education",
                "experience",
                "skills",
                "contact",
                "email",
                "phone",
            ]

            if not any(word in lower for word in excluded):
                return clean_line

    # Fallback: filename
    name = Path(filename).stem
    name = re.sub(r"[_\-]+", " ", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name if name else "Unknown Candidate"


def extract_skills(text):
    """Extract commonly used technical skills."""

    skills = [
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
        "postgresql",
        "mongodb",
        "git",
        "github",
        "docker",
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
        "rag",
        "langchain",
        "streamlit",
        "tensorflow",
        "pytorch",
        "pandas",
        "numpy",
        "power bi",
        "excel",
        "figma",
        "ui/ux",
        "ui ux",
        "data analytics",
        "data analysis",
        "faiss",
        "rest api",
        "api",
    ]

    text_lower = text.lower()

    found = []

    for skill in skills:
        if skill in text_lower:
            found.append(skill)

    return sorted(set(found))


# ============================================================
# AI / RAG COMPONENTS
# ============================================================

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


# ============================================================
# PDF PROCESSING
# ============================================================

def process_resumes(uploaded_files):

    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS

    if not uploaded_files:
        return False

    progress = st.progress(0)
    status_text = st.empty()

    try:

        # ----------------------------------------------------
        # STEP 1 - READ PDF FILES
        # ----------------------------------------------------

        status_text.info("📖 Step 1/5 — Reading resume files...")

        all_documents = []
        resume_texts = {}
        candidate_names = {}

        total_files = len(uploaded_files)

        for index, uploaded_file in enumerate(uploaded_files):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            try:

                loader = PyPDFLoader(temp_path)
                docs = loader.load()

                full_text = "\n".join(
                    doc.page_content
                    for doc in docs
                )

                full_text = clean_text(full_text)

                filename = uploaded_file.name

                resume_texts[filename] = full_text

                candidate_names[filename] = extract_name(
                    full_text,
                    filename
                )

                for doc in docs:
                    doc.metadata["source"] = filename
                    doc.metadata["candidate"] = candidate_names[filename]

                all_documents.extend(docs)

            finally:

                try:
                    os.remove(temp_path)
                except Exception:
                    pass

            progress.progress(
                int(((index + 1) / total_files) * 20)
            )

        if not all_documents:
            status_text.error(
                "❌ No readable content was found in the uploaded PDFs."
            )
            return False

        # ----------------------------------------------------
        # STEP 2 - SPLIT DOCUMENTS
        # ----------------------------------------------------

        status_text.info(
            "✂️ Step 2/5 — Preparing resume text..."
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(all_documents)

        if not chunks:
            status_text.error(
                "❌ Could not create text chunks from the resumes."
            )
            return False

        progress.progress(40)

        # ----------------------------------------------------
        # STEP 3 - LOAD EMBEDDING MODEL
        # ----------------------------------------------------

        status_text.info(
            "🧠 Step 3/5 — Loading AI search model..."
        )

        embeddings = get_embeddings()

        progress.progress(60)

        # ----------------------------------------------------
        # STEP 4 - CREATE FAISS DATABASE
        # ----------------------------------------------------

        status_text.info(
            "🔎 Step 4/5 — Creating searchable resume database..."
        )

        vectorstore = FAISS.from_documents(
            chunks,
            embeddings
        )

        progress.progress(85)

        # ----------------------------------------------------
        # STEP 5 - SAVE RESULTS
        # ----------------------------------------------------

        status_text.info(
            "💾 Step 5/5 — Saving processed resumes..."
        )

        st.session_state.documents = all_documents
        st.session_state.vectorstore = vectorstore
        st.session_state.resume_texts = resume_texts
        st.session_state.candidate_names = candidate_names
        st.session_state.processed_files = [
            uploaded_file.name
            for uploaded_file in uploaded_files
        ]
        st.session_state.chunk_count = len(chunks)

        st.session_state.match_results = []
        st.session_state.chat_history = []

        progress.progress(100)

        status_text.success(
            f"✅ Successfully processed {len(uploaded_files)} resume(s)."
        )

        return True

    except Exception as e:

        status_text.error(
            "❌ Resume processing failed."
        )

        st.exception(e)

        return False


# ============================================================
# JOB MATCHING
# ============================================================

def calculate_match(resume_text, job_description):

    resume_skills = set(
        extract_skills(resume_text)
    )

    job_skills = set(
        extract_skills(job_description)
    )

    if not job_skills:
        return 0, [], []

    matched = sorted(
        resume_skills.intersection(job_skills)
    )

    missing = sorted(
        job_skills.difference(resume_skills)
    )

    score = round(
        (len(matched) / len(job_skills)) * 100
    )

    return score, matched, missing


def run_job_matching(job_description):

    results = []

    for filename, resume_text in st.session_state.resume_texts.items():

        score, matched, missing = calculate_match(
            resume_text,
            job_description
        )

        candidate_name = st.session_state.candidate_names.get(
            filename,
            Path(filename).stem
        )

        results.append({
            "Candidate": candidate_name,
            "Resume": filename,
            "Match %": score,
            "Matched Skills": ", ".join(matched) if matched else "None",
            "Missing Skills": ", ".join(missing) if missing else "None"
        })

    # Sort by score for display
    results.sort(
        key=lambda x: x["Match %"],
        reverse=True
    )

    st.session_state.match_results = results

    return results


# ============================================================
# AI RESUME CHAT
# ============================================================

def ask_resume(question):

    if st.session_state.vectorstore is None:
        return "Please upload and process resumes first."

    llm = get_llm()

    if llm is None:
        return (
            "Google Gemini API key is not configured. "
            "Please add GOOGLE_API_KEY in your environment variables."
        )

    try:

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={
                "k": 5
            }
        )

        relevant_docs = retriever.invoke(question)

        if not relevant_docs:
            return "I could not find relevant information in the uploaded resumes."

        context_parts = []

        for doc in relevant_docs:

            candidate = doc.metadata.get(
                "candidate",
                "Unknown Candidate"
            )

            source = doc.metadata.get(
                "source",
                "Unknown Resume"
            )

            context_parts.append(
                f"""
Candidate: {candidate}
Resume: {source}

Content:
{doc.page_content}
"""
            )

        context = "\n\n".join(context_parts)

        prompt = f"""
You are ResumeAI, an AI assistant for analyzing uploaded resumes.

Answer the user's question using ONLY the resume information provided below.

If the information is not available in the resumes, clearly say that
the information was not found.

Do not invent candidate information.

Resume Context:
{context}

User Question:
{question}

Give a clear and concise answer.
"""

        response = llm.invoke(prompt)

        return response.content

    except Exception as e:

        return f"AI response failed: {str(e)}"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📄 ResumeAI")

    st.caption(
        "Intelligent Resume Analysis"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Upload Resumes",
            "Job Match",
            "Compare Candidates",
            "AI Resume Chat"
        ]
    )

    st.divider()

    st.subheader("System Status")

    if st.session_state.processed_files:

        st.success(
            f"{len(st.session_state.processed_files)} resume(s) ready"
        )

    else:

        st.info(
            "No resumes processed"
        )

    if GOOGLE_API_KEY:
        st.success("Gemini API configured")
    else:
        st.warning("Gemini API key not configured")


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("📊 Resume Intelligence")

    st.write(
        "Search, analyze, compare and understand candidates "
        "using Retrieval-Augmented Generation."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📄 Resumes",
            len(st.session_state.processed_files)
        )

    with col2:
        st.metric(
            "🧩 Text Chunks",
            st.session_state.chunk_count
        )

    with col3:
        st.metric(
            "👤 Candidates",
            len(st.session_state.candidate_names)
        )

    with col4:
        if st.session_state.vectorstore:
            status = "Ready"
        else:
            status = "Not Ready"

        st.metric(
            "🔎 AI Search",
            status
        )

    st.divider()

    st.subheader("📄 Resume Analysis")

    with st.container(border=True):

        st.write(
            "Upload PDF resumes and convert them into a searchable "
            "knowledge base using embeddings and FAISS."
        )

        st.write(
            "You can then use the Job Match, Compare Candidates "
            "and AI Resume Chat sections."
        )

    st.subheader("⚙️ How it works")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("### 1️⃣")
        st.write("Upload PDF resumes")

    with step2:
        st.markdown("### 2️⃣")
        st.write("Extract and split text")

    with step3:
        st.markdown("### 3️⃣")
        st.write("Create embeddings")

    with step4:
        st.markdown("### 4️⃣")
        st.write("Search with AI")

    if not st.session_state.processed_files:

        st.info(
            "👈 Go to **Upload Resumes** from the sidebar to begin."
        )


# ============================================================
# UPLOAD RESUMES
# ============================================================

elif page == "Upload Resumes":

    st.title("📤 Upload Resumes")

    st.write(
        "Upload one or more PDF resumes to build your searchable "
        "resume database."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF resume files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write(
            f"**{len(uploaded_files)} file(s) selected**"
        )

        for file in uploaded_files:

            st.write(
                f"📄 {file.name} — "
                f"{file.size / 1024:.1f} KB"
            )

        st.divider()

        if st.button(
            "🚀 Process Resumes",
            type="primary",
            use_container_width=True
        ):

            with st.status(
                "Processing resumes...",
                expanded=True
            ):

                success = process_resumes(
                    uploaded_files
                )

                if success:

                    st.success(
                        "Resume processing completed successfully!"
                    )

        st.divider()

        if st.session_state.processed_files:

            st.subheader("Previously Processed Resumes")

            for filename in st.session_state.processed_files:

                candidate = st.session_state.candidate_names.get(
                    filename,
                    "Unknown"
                )

                st.write(
                    f"✅ **{candidate}** — {filename}"
                )

    else:

        st.info(
            "Upload PDF resumes above to start."
        )


# ============================================================
# JOB MATCH
# ============================================================

elif page == "Job Match":

    st.title("🎯 Job Match")

    st.write(
        "Enter a job description to compare it with the "
        "processed resumes."
    )

    if not st.session_state.resume_texts:

        st.warning(
            "Please upload and process resumes first."
        )

        st.stop()

    job_description = st.text_area(
        "Job Description",
        height=250,
        placeholder=(
            "Example:\n"
            "We are looking for a Java Developer with "
            "Spring Boot, SQL, Git and REST API experience."
        )
    )

    if st.button(
        "🔍 Analyze Job Match",
        type="primary"
    ):

        if not job_description.strip():

            st.warning(
                "Please enter a job description."
            )

        else:

            with st.spinner(
                "Analyzing resumes..."
            ):

                results = run_job_matching(
                    job_description
                )

            st.success(
                f"Analyzed {len(results)} resume(s)."
            )

    if st.session_state.match_results:

        st.divider()

        st.subheader("Match Results")

        for result in st.session_state.match_results:

            with st.container(border=True):

                col1, col2 = st.columns(
                    [4, 1]
                )

                with col1:

                    st.subheader(
                        f"👤 {result['Candidate']}"
                    )

                    st.caption(
                        result["Resume"]
                    )

                with col2:

                    st.metric(
                        "Match",
                        f"{result['Match %']}%"
                    )

                st.write(
                    "**Matched Skills:**"
                )

                if result["Matched Skills"] != "None":
                    st.write(
                        result["Matched Skills"]
                    )
                else:
                    st.write("None found")

                st.write(
                    "**Missing Skills:**"
                )

                if result["Missing Skills"] != "None":
                    st.write(
                        result["Missing Skills"]
                    )
                else:
                    st.write("None")


# ============================================================
# COMPARE CANDIDATES
# ============================================================

elif page == "Compare Candidates":

    st.title("👥 Compare Candidates")

    if not st.session_state.resume_texts:

        st.warning(
            "Please upload and process resumes first."
        )

        st.stop()

    rows = []

    for filename, resume_text in st.session_state.resume_texts.items():

        candidate = st.session_state.candidate_names.get(
            filename,
            Path(filename).stem
        )

        skills = extract_skills(
            resume_text
        )

        rows.append({
            "Candidate": candidate,
            "Resume": filename,
            "Skills": ", ".join(skills),
            "Skill Count": len(skills),
            "Characters": len(resume_text)
        })

    st.write(
        f"**{len(rows)} candidate(s) available**"
    )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Candidate Details")

    candidate_names = [
        row["Candidate"]
        for row in rows
    ]

    selected_candidate = st.selectbox(
        "Select a candidate",
        candidate_names
    )

    selected_row = next(
        row
        for row in rows
        if row["Candidate"] == selected_candidate
    )

    with st.container(border=True):

        st.subheader(
            f"📄 {selected_row['Candidate']}"
        )

        st.write(
            f"**Resume:** {selected_row['Resume']}"
        )

        st.write(
            f"**Number of detected skills:** "
            f"{selected_row['Skill Count']}"
        )

        st.write(
            f"**Resume text length:** "
            f"{selected_row['Characters']} characters"
        )

        st.write(
            "**Detected Skills:**"
        )

        if selected_row["Skills"]:
            st.write(
                selected_row["Skills"]
            )
        else:
            st.write(
                "No predefined skills detected."
            )


# ============================================================
# AI RESUME CHAT
# ============================================================

elif page == "AI Resume Chat":

    st.title("🤖 AI Resume Chat")

    st.write(
        "Ask questions about the uploaded resumes using "
        "Retrieval-Augmented Generation."
    )

    if st.session_state.vectorstore is None:

        st.warning(
            "Please upload and process resumes before using AI Resume Chat."
        )

        st.stop()

    if not GOOGLE_API_KEY:

        st.error(
            "GOOGLE_API_KEY is not configured."
        )

        st.info(
            "Add GOOGLE_API_KEY to the Render Environment Variables."
        )

        st.stop()

    # Display previous messages

    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    question = st.chat_input(
        "Ask something about the resumes..."
    )

    if question:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching resumes..."
            ):

                answer = ask_resume(
                    question
                )

            st.write(answer)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    if st.session_state.chat_history:

        st.divider()

        if st.button(
            "🗑️ Clear Chat"
        ):

            st.session_state.chat_history = []

            st.rerun()
