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

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))  # type: ignore

# get all text from all pdf
def get_pdf_into_text(pdf_docs:list) -> str:
    text = ''
    
    # looping on each pdf file
    
    for pdf in pdf_docs:
        try:
            pdf_content=''
            reader = PdfReader(pdf)
            for page in reader.pages:
                pdf_content += page.extract_text()
            print(pdf_content[:1000])
            text += pdf_content
        except Exception as e:
            CustomException(e,sys)
            logger.error("Not able to extract text from pdf:",pdf)
    
    return text


# now converting text into chunks
def create_chunks(text) -> list:
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
    try:  
        # documents = [Document(page_content=chunk) for chunk in chunks]
        
        # converting words/characters in vector(number)
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        # vector_store = Chroma.from_documents(
        #     documents=documents,
        #     embedding=embedding,
        #     persist_directory='./chroma_index'
        #     )
        # vector_store.persist()
        
        # using FAISS
        vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=embedding
        )
        vector_store.save_local("faiss_index")
        
        vector_store.save_local("faiss_index")
    except Exception as e:
        CustomException(e,sys)
        logger.error("Error while creating vector store")

def get_qa_chain():
    
    try:
        prompt_template = """
        You are a helpful and precise AI assistant. You answer questions strictly based on the context provided.
        
        If the answer is not present in the context, reply with:
        'Answer is not available in that context.' or 'Not able to find the answer from PDFs.'

        If the question is unrelated to the context, reply with:
        'Not able to find the answer from PDFs.'

        Do not add any extra information. Only use the provided context to generate the answer. Be concise, accurate, and to the point.
        You must only answer using the provided context.
        Do not guess or hallucinate. Be brief and factual.

        Context:
        {context}

        Question:
        {question}

        Answer:
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

def get_answer_from_chain(question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectordb = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    
    # Use top 5 results for better context
    docs = vectordb.similarity_search(question, k=5)
    
    chain = get_qa_chain()
    
    response = chain.invoke({ # type: ignore
        'input_documents': docs,
        'question': question
    })
    
    # print(response)
    
    if isinstance(response, dict):
        return response.get('output_text', response.get('text', ''))
    
    return response
  