from dotenv import load_dotenv
import sys
import os
from utils import logger,CustomException
# from langchain.schema import Document
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings,ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
# from langchain.vectorstores import 
from langchain.chains.question_answering import load_qa_chain  
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.vectorstores.utils import cosine_similarity # type: ignore
import hashlib
import json
from datetime import datetime



import warnings
warnings.filterwarnings("ignore")

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))  # type: ignore
query_cache = [] 


def chunk_id(text: str):
    return hashlib.md5(text.strip().encode('utf-8')).hexdigest()

# get all text from all pdf
def get_pdf_into_text(docs:list) -> str:
    text = ''
    
    # looping on each pdf file

    logger.info("Extracting text from PDF:")
    for doc in docs:
        try:
            content=''
            if(str(doc).endswith('.pdf')):
                reader = PdfReader(doc)
                for page in reader.pages:
                    content += page.extract_text()
            elif(str(doc).endswith('.txt')):
                with open(doc, 'r') as f:
                    content = f.read()
            # print(content[:1000])
            text += content
        except Exception as e:
            CustomException(e,sys)
            logger.error("Not able to extract text from pdf:",doc)
    
    return text


def hash_question(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode('utf-8')).hexdigest()


def get_pdf_into_text_for_streamlit(docs: list) -> str:
    text = ''
    logger.info("Extracting text from PDF:")
    filename=''
    for doc in docs:
        try:
            content = ''
            filename = doc.name if hasattr(doc, 'name') else str(doc)

            if filename.endswith('.pdf'):
                reader = PdfReader(doc)
                for page in reader.pages:
                    content += page.extract_text() or ""
            elif filename.endswith('.txt'):
                content = doc.read().decode("utf-8")
            text += content
        except Exception as e:
            logger.error(f"Failed to extract text from: {filename}")
            raise CustomException(e, sys)
    return text


# now converting text into chunks
def create_chunks(text) -> list:
    logger.info("Creating text chunks:")
    res = []
    try:
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        length_function=len
    )
        res = text_splitter.split_text(text) 
    except Exception as e:
        CustomException(e,sys)
        logger.error("Error while doing text splitting")
    finally:

        return res

def get_vector_store(chunks):
    logger.info("Creating vector store")
    try:
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        # Convert each chunk into a Document with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": f"chunk_{i}",
                    "chunk_id": chunk_id(chunk)  # Add a unique ID
                }
            )
            for i, chunk in enumerate(chunks)
        ]

        vector_store = FAISS.from_documents(documents, embedding)
        vector_store.save_local("faiss_index")
    except Exception as e:
        CustomException(e, sys)
        logger.error("Error while creating vector store")

def get_qa_chain():
    logger.info("Creating QA Chain")
    try:
        prompt_template = """
        You are a helpful and precise AI assistant. You answer questions strictly based on the context provided.
        
        If the answer is not present in the context, reply with:
        'Answer is not available in that context.' or 'Not able to find the answer from PDFs.'
        
        At the end of your answer, mention the source 

        If the question is unrelated to the context, reply with:
        'Not able to find the answer from PDFs.'

        Do not add any extra information. Only use the provided context to generate the answer. Be concise, accurate, and to the point.
        You must only answer using the provided context.
        Do not guess or hallucinate. Be brief and factual.

        Context:
        {context}

        Question:
        {question}

      
        """
        
        model = ChatGoogleGenerativeAI(
            model='gemini-1.5-flash',
            temperature=0.3
        )
        
        prompt= PromptTemplate(
            template=prompt_template,
            input_variables=['context','question']
        )
        
        chain = load_qa_chain(
            llm=model,
            chain_type='stuff',
            prompt=prompt
        )
        
        return chain

    except Exception as e:
        CustomException(e,sys)
        logger.error("Error while loading QA Chain")
        return None

def save_feedback(feedback_data, feedback_file="feedback_log.json"):
    try:
        dir = feedback_file.split(".")[0]
        os.makedirs(dir, exist_ok=True)
        feedback_file = os.path.join(dir, feedback_file)
        if os.path.exists(feedback_file):
            with open(feedback_file, "r") as f:
                data = json.load(f)
        else:
            data = []

        feedback_data["timestamp"] = datetime.utcnow().isoformat() + "Z"
        data.append(feedback_data)

        with open(feedback_file, "w") as f:
            json.dump(data, f, indent=4)

        logger.info("Full feedback saved successfully.")
        
    except Exception as e:
        logger.error("Failed to save full feedback.")
        raise CustomException(e, sys)


def get_answer_from_chain(question):
    logger.info(f"Getting the answer to the user question:\n{question}")
    similarity_threshold = 0.9
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    question_embedding = embedding_model.embed_query(question)

    # Check for similar question in cache
    for cached in query_cache:
        sim = cosine_similarity([question_embedding], [cached["embedding"]])[0][0]
        if sim >= similarity_threshold:
            logger.info(f"Found semantically similar question in cache (similarity={sim:.2f}).")
            return cached["result"]

    # If not found, proceed normally
    vectordb = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
    docs = vectordb.similarity_search(question, k=5)
    chunk_ids = [doc.metadata.get("chunk_id", "unknown") for doc in docs]

    chain = get_qa_chain()

    response = chain.invoke({ # type: ignore
        'input_documents': docs,
        'question': question
    })

    answer = response.get('output_text', response.get('text', '')).strip()
    sources = [doc.metadata.get("source", "unknown") for doc in docs]

    result = {
        "answer": answer,
        "sources": sources,
        "chunk_ids": chunk_ids
    }

    # Save to cache
    query_cache.append({
        "question": question,
        "embedding": question_embedding,
        "result": result
    })
    logger.info("New result cached successfully.")

    return result
