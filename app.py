import streamlit as st
import sys
import time
from utils import logger, CustomException
from utils.common import get_pdf_into_text_for_streamlit, create_chunks, get_vector_store, get_answer_from_chain

st.set_page_config(
    page_title="Multi PDF/ TEXT Reader",
    page_icon=":books:",
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title("📚 Chat with Multiple PDFs/ Texts using AI")


with st.sidebar:
    st.header("📄 Upload your PDFs/ Texts")
    pdf_docs = st.file_uploader("Upload PDF/ Text files", type=["pdf", "txt"], accept_multiple_files=True)
    # process_btn = st.button("Process PDFs")

    if st.button("Submit & Process"):
        with st.spinner("Processing the data......"):
            raw_text = get_pdf_into_text_for_streamlit(pdf_docs)
            text_chunks = create_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("PDFs Are Processed Successfully!")

userquestion = st.text_input("Ask a question about the PDFs:")

if userquestion and userquestion!="":
    with st.spinner("Finding the answer..."):
        try:
            now=time.time()
            answer=get_answer_from_chain(userquestion)
            st.write(answer)
            st.write(time.time()-now)
        except Exception as e:
            CustomException(e,sys)
            logger.error("Error while questioning with llm")
            