import streamlit as st
import sys
import time
from datetime import datetime
from utils import logger, CustomException
from utils.common import (
    save_feedback,
    get_pdf_into_text_for_streamlit,
    create_chunks,
    get_vector_store,
    get_answer_from_chain
)

# Page Config
st.set_page_config(
    page_title="Multi PDF/ TEXT Reader",
    page_icon="📚",
    layout='wide',
    initial_sidebar_state='expanded'
)

# Title
st.title("📚 Chat with Multiple PDFs/ Texts using AI")

# Sidebar for Uploads
with st.sidebar:
    st.header("📄 Upload your PDFs/ Texts")
    pdf_docs = st.file_uploader(
        "Upload PDF/ Text files",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )

    if st.button("Submit & Process"):
        with st.spinner("Processing the data......"):
            raw_text = get_pdf_into_text_for_streamlit(pdf_docs)
            text_chunks = create_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDFs Are Processed Successfully!")

# Main Question Input
userquestion = st.text_input("Ask a question about the PDFs:")

if userquestion and userquestion.strip() != "":
    with st.spinner("Finding the answer..."):
        try:
            start_time = time.time()
            response = get_answer_from_chain(userquestion)
            answer = response['answer']
            st.write(answer)
            st.divider()
            st.write(f"⏱️ Response time: {time.time() - start_time:.2f} seconds")

            # Feedback Title
            # Feedback Title
            st.markdown("### Was this answer helpful?")

            # Display Like and Dislike buttons side-by-side
            col1, col2 = st.columns([1, 1])

            with col1:
                like_clicked = st.button("👍 Like", key="like_button")

            with col2:
                dislike_clicked = st.button("👎 Dislike", key="dislike_button")

            if like_clicked:
                save_feedback({
                    "question": userquestion,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "feedback": "positive"
                })
                st.success("✅ Thanks for your positive feedback!")

            if dislike_clicked:
                save_feedback({
                    "question": userquestion,
                    "answer": answer,
                    "timestamp": datetime.now().isoformat(),
                    "feedback": "negative"
                })
                st.warning("⚠️ Thanks for your feedback. We'll use it to improve!")


        except Exception as e:
            CustomException(e, sys)
            logger.error("Error while questioning with LLM")
            st.error("An error occurred while processing your question.")
