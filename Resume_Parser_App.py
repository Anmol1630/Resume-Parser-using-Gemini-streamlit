# installatoin
# pip install langchain_openai langchain-google-genai python-dotenv streamlit
# pip install -U langchain-community

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import json


from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader,
    TextLoader
)
from langchain_core.prompts import PromptTemplate


# Step 2: Config / LLM

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

PROMPT_TEMPLATE = """
You are an expert resume parser. Given the resume text, extract the following fields and return a single valid JSON object:

{{"Name": "...",
  "LinkedIn": "...",
  "Skills": [...],
  "Education": [...],
  "Experience": [...],
  "Projects": [...],
}}

Rules:
- If a field cannot be found, set its value to "No idea".
- Return ONLY valid JSON (no extra commentary).
- Keep lists as arrays, and keep Experience/Projects as arrays of short strings.

Resume text:
{text}
"""


prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["text"])


# Step 3: Helpers

def load_resume_docs(uploaded_file):
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(temp_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(temp_path)
    elif uploaded_file.name.endswith(".txt"):
        loader = TextLoader(temp_path)
    else:
        return None
    return loader.load()




# Step 4: Streamlit UI

def main():
    st.set_page_config(
        page_title="Resume Parser",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # css
    st.markdown("""
        <style>
        body {
            background: linear-gradient(135deg, #1E1E2F, #2E2E47);
            color: white;
            font-family: 'Inter', sans-serif;
        }
        .main {
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(12px);
            padding: 2rem;
            border-radius: 1.5rem;
            box-shadow: 0 8px 30px rgba(0,0,0,0.2);
        }
        h1, h2, h3, h4 {
            color: #E6E6E6;
        }
        .stTextInput, .stTextArea, .stFileUploader {
            border-radius: 10px !important;
        }
        .stButton>button {
            background: linear-gradient(90deg, #6a11cb, #2575fc);
            color: white;
            border: none;
            padding: 0.8em 1.5em;
            border-radius: 10px;
            font-weight: 600;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            box-shadow: 0px 4px 20px rgba(37,117,252,0.4);
        }
        </style>
    """, unsafe_allow_html=True)

    # 💎 Header
    st.markdown("<h1 style='text-align:center;'>📄 Resume Parser</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:1.1em; color:#ccc;'>AI-powered assistant to help HRs shortlist candidates faster</p>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #444;'>", unsafe_allow_html=True)

    # 📁 File Upload Section
    with st.container():
        st.markdown("### 🚀 Upload Your Resume")
        uploaded_file = st.file_uploader(
            "Drop your resume here (PDF, DOCX, or TXT)",
            type=["pdf", "docx", "txt"],
            help="Upload the candidate resume to extract key details using AI"
        )

    # If a file is uploaded
    if uploaded_file:
        with st.spinner("🧠 Reading and parsing your resume..."):
            docs = load_resume_docs(uploaded_file)
            if not docs:
                st.error("⚠️ Unsupported file type. Please upload a PDF, DOCX, or TXT.")
                return

        st.success("✅ Resume successfully loaded!")
        st.markdown("<hr>", unsafe_allow_html=True)

        # 📝 Preview Section
        st.markdown("### 🧾 Extracted Resume Preview")
        preview_text = "\n\n".join([d.page_content for d in docs])[:4000]
        st.text_area(
            "Preview (First 4000 characters)",
            value=preview_text,
            height=250,
            label_visibility="collapsed"
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # 🎯 Parse Button
        st.markdown("### 🤖 Generate Candidate Insights")
        st.caption("Click below to extract key info like skills, experience, and education.")

        if st.button("✨ Parse with AI", use_container_width=True):
            with st.spinner("🔍 Analyzing resume using LLM..."):
                full_text = "\n\n".join([d.page_content for d in docs])
                formatted_prompt = prompt.format(text=full_text)

                response = llm.invoke(formatted_prompt)
                try:
                    parsed_json = json.loads(response.content)
                    st.success("🎉 Parsing Complete! Here are the extracted insights:")
                    st.json(parsed_json)
                except json.JSONDecodeError:
                    st.warning("⚠️ Could not parse JSON properly. Here is the raw output:")
                    st.write(response.content)

    # 🧡 Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='text-align: center; color: #aaa; font-size: 0.9em;'>
            Made with 💡 by <b>Anmol Prashar</b> | Powered by <b>LangChain + Streamlit</b>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()