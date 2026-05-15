import streamlit as st
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer, util
import torch
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="College Admission Chatbot",
    layout="centered"
)

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.title("🎓 About")

st.sidebar.info("""
College Admission Chatbot

• Fine-Tuned LLM (LoRA)
• Semantic Search
• Built with Streamlit
""")

dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=False)

# ---------------------------
# PREMIUM CSS
# ---------------------------
st.markdown("""
<style>

/* Animated Background */
body {
    background: linear-gradient(-45deg, #e3f2fd, #ffffff, #f1f8e9, #e8f5e9);
    background-size: 400% 400%;
    animation: gradient 12s ease infinite;
}

@keyframes gradient {
    0% {
        background-position: 0% 50%;
    }
    50% {
        background-position: 100% 50%;
    }
    100% {
        background-position: 0% 50%;
    }
}

/* Chat bubbles */
.chat-bubble {
    padding: 14px 18px;
    border-radius: 16px;
    margin-bottom: 12px;
    max-width: 80%;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    transition: 0.3s ease;
    font-size: 16px;
}

.chat-bubble:hover {
    transform: scale(1.02);
}

.user-bubble {
    background-color: #DCF8C6;
    margin-left: auto;
}

.bot-bubble {
    background-color: #F1F0F0;
    margin-right: auto;
}

/* Buttons */
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border: none;
    transition: 0.3s;
}

div.stButton > button:first-child:hover {
    background-color: #45a049;
    transform: scale(1.03);
}

/* Footer */
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f9f9f9;
    color: gray;
    text-align: center;
    padding: 10px;
    font-size: 14px;
    border-top: 1px solid #ddd;
}

/* Mobile responsive */
@media screen and (max-width: 768px) {
    .chat-bubble {
        max-width: 95%;
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# PROFESSIONAL DARK MODE
# ---------------------------
if dark_mode:

    st.markdown("""
    <style>

    /* Whole app */
    .stApp {
        background-color: #020817 !important;
        color: white;
    }

    /* Main app container */
    [data-testid="stAppViewContainer"] {
        background-color: #020817 !important;
    }

    /* Header */
    [data-testid="stHeader"] {
        background-color: #020817 !important;
    }

    /* Toolbar */
    [data-testid="stToolbar"] {
        background-color: #020817 !important;
    }

    /* Main block */
    .block-container {
        background-color: #020817 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0F172A !important;
    }

    /* Titles */
    h1, h2, h3 {
        color: #4ADE80 !important;
    }

    /* Text */
    p, label, div {
        color: #E2E8F0 !important;
    }

    /* Chat bubbles */
    .chat-bubble {
        background-color: #1E293B !important;
        color: white !important;
        border: 1px solid #334155;
    }

    .user-bubble {
        background-color: #166534 !important;
        color: white !important;
    }

    .bot-bubble {
        background-color: #1E293B !important;
        color: white !important;
    }

    /* Input box */
    input {
        background-color: #1E293B !important;
        color: white !important;
    }

    /* Buttons */
    div.stButton > button:first-child {
        background-color: #22C55E !important;
        color: white !important;
    }

    /* Footer */
    .footer {
        background-color: #0F172A !important;
        color: #E2E8F0 !important;
    }

    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# TITLE
# ---------------------------

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
    Domain-Specific Intelligent Chatbot using Fine-Tuned LLM
    </h1>
    
    """,
    unsafe_allow_html=True
)

# ---------------------------
# HERO SECTION
# ---------------------------
st.markdown("""
<div style='text-align:center; padding:20px;'>

<h3>🤖 AI-Powered College Admission Assistant</h3>

<p style='font-size:18px; color:gray;'>
Get instant answers about admissions, fees, courses,
scholarships, and campus facilities.
</p>

</div>
""", unsafe_allow_html=True)

# ---------------------------
# FEATURE CARDS
# ---------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.info("🧠 Fine-Tuned LLM")

with col2:
    st.info("🔍 Semantic Search")

with col3:
    st.info("⚡ Real-Time Responses")

# ---------------------------
# PDF UPLOAD SECTION
# ---------------------------
st.write("## 📄 Upload College PDF")

uploaded_pdf = st.file_uploader(
    "Upload brochure / admission PDF",
    type="pdf"
)

pdf_text = ""

# ---------------------------
# READ PDF
# ---------------------------
if uploaded_pdf is not None:

    st.success("✅ PDF uploaded successfully!")

    pdf_reader = PdfReader(uploaded_pdf)

    for page in pdf_reader.pages:

        text = page.extract_text()

        if text:
            pdf_text += text

# ---------------------------
# CHUNKING
# ---------------------------
vector_db = None

if pdf_text:

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(pdf_text)

# ---------------------------
# CREATE VECTOR DATABASE
# ---------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_texts(
        chunks,
        embeddings
    )
# ---------------------------
# SESSION STATE
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "temp_input" not in st.session_state:
    st.session_state.temp_input = ""

# ---------------------------
# TRY ASKING SECTION
# ---------------------------
st.write("### 💡 Try asking:")

cols = st.columns(3)

questions = [
    "What is eligibility for admission?",
    "What courses are offered?",
    "Is hostel available?"
]

for i, q in enumerate(questions):

    if cols[i].button(q, key=f"suggest_{i}"):

        st.session_state.input_value = q

        st.rerun()

# ---------------------------
# LOAD DATASET
# ---------------------------
@st.cache_resource
def load_data():

    with open("data.json") as f:
        return json.load(f)

data = load_data()

# ---------------------------
# LOAD SEMANTIC MODEL
# ---------------------------
@st.cache_resource
def load_similarity_model(data):

    model_sim = SentenceTransformer('all-MiniLM-L6-v2')

    questions = [item["instruction"] for item in data]

    embeddings = model_sim.encode(
        questions,
        convert_to_tensor=True
    )

    return model_sim, embeddings

model_sim, question_embeddings = load_similarity_model(data)

# ---------------------------
# LOAD FINE-TUNED MODEL
# ---------------------------
@st.cache_resource
def load_model():

    base = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(base)

    model = AutoModelForCausalLM.from_pretrained(base)

    model = PeftModel.from_pretrained(
        model,
        "fine_tuned_model"
    )

    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------
# CLEAR INPUT FUNCTION
# ---------------------------
def clear_text():

    st.session_state.input_value = ""

    st.rerun()

# ---------------------------
# CHAT FUNCTION
# ---------------------------
def chat(query):

# ---------------------------
# SEMANTIC SEARCH
# ---------------------------
    q_emb = model_sim.encode(
        query,
        convert_to_tensor=True
    )

    scores = util.cos_sim(
        q_emb,
        question_embeddings
    )[0]

    idx = int(torch.argmax(scores))

    score = float(scores[idx])

    # Exact / Similar Match
    if score > 0.6:
        return data[idx]["response"]

# ---------------------------
# PDF RAG SEARCH
# ---------------------------
    if vector_db is not None:

        docs = vector_db.similarity_search(query, k=2)

        if docs:

            pdf_answer = docs[0].page_content

            return f"📄 From Uploaded PDF:\n\n{pdf_answer}"

# ---------------------------
# UNKNOWN QUESTION HANDLING
# ---------------------------
    if score < 0.4:

        return "I'm not sure about that. Please ask questions related to college admissions."

# ---------------------------
# LLM FALLBACK
# ---------------------------
    prompt = f"### Instruction:\n{query}\n\n### Response:\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=80,
        temperature=0.5,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "### Response:" in text:
        return text.split("### Response:")[-1].strip()

    return text

# ---------------------------
# PDF RAG SEARCH
# ---------------------------
    if vector_db is not None:

        docs = vector_db.similarity_search(query, k=2)

        if docs:

            pdf_answer = docs[0].page_content

            return f"📄 From Uploaded PDF:\n\n{pdf_answer}"

    # LLM Fallback
    prompt = f"### Instruction:\n{query}\n\n### Response:\n"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=80,
        temperature=0.5,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    if "### Response:" in text:
        return text.split("### Response:")[-1].strip()

    return text

# ---------------------------
# CHAT INPUT FORM
# ---------------------------
with st.form("chat_form", clear_on_submit=True):

    col1, col2 = st.columns([8, 1])

    with col1:

        user_query = st.text_input(
            "💬 Ask your question:",
            placeholder="Type your question here..."
        )

    with col2:

        send = st.form_submit_button("Send")

# ---------------------------
# PROCESS MESSAGE
# ---------------------------
if send and user_query:

    with st.spinner("🤖 AI is analyzing your query..."):

        response = chat(user_query)

    # Save messages
    st.session_state.messages.append(
        ("user", user_query)
    )

    st.session_state.messages.append(
        ("bot", response)
    )

    st.rerun()
# ---------------------------
# DISPLAY CHAT
# ---------------------------
for role, msg in st.session_state.messages:

    if role == "user":

        st.markdown(
            f"""
            <div class="chat-bubble user-bubble">
            🧑 {msg}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
            <div class="chat-bubble bot-bubble">
            🤖 {msg}
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------------------
# CLEAR CHAT BUTTON
# ---------------------------
if st.button(
    "Clear Chat",
    key="clear_chat_btn"
):

    st.session_state.messages = []

    st.session_state.temp_input = ""

    st.rerun()

# ---------------------------
# SPACING
# ---------------------------
st.markdown("<br><br><br>", unsafe_allow_html=True)

# ---------------------------
# FIXED FOOTER
# ---------------------------
st.markdown("""
<div class="footer">
🚀 Built by Ugesh Reddy | ML Internship Project
</div>
""", unsafe_allow_html=True)