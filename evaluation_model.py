import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from utils.common import get_pdf_into_text, create_chunks, get_vector_store, get_answer_from_chain
from utils import logger
import google.generativeai as genai


# Load API key from .env
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY")) # type: ignore

# Setup embedding
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Similarity Functions ---
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def similarity_score(a, b):
    a_vector = embedding.embed_query(a)
    b_vector = embedding.embed_query(b)
    return cosine_similarity(a_vector, b_vector)

def embedding_based_completeness(expected, actual):
    expected_sentences = expected.split('.')
    covered = 0

    for sent in expected_sentences:
        if sent.strip():
            sim = similarity_score(sent.strip(), actual)
            if sim > 0.75:
                covered += 1

    return covered / len(expected_sentences) if expected_sentences else 0


# --- Evaluation Dataset ---
evaluation_data = {
    "ArtificialIntelligenceWhatisit-RiturajMahato.pdf": {
        "question": "What are the key benefits and concerns of using Artificial Intelligence in education?",
        "answer": "Key benefits include personalized learning, reduced teacher workload, intelligent teaching tools, and flexibility through platforms like Coursera and Udemy. Major concerns involve potential job loss, ethical risks, overdependence on technology, and lack of sociological considerations in implementation."
    },
    
#     "Ch1Materialpdf__2025_06_11_13_08_19.pdf": {
#         "question": "What are the key terminologies in Reinforcement Learning and what issues are commonly faced in Machine Learning?",
#         "answer": "Key terms include Agent, Environment, Action, Reward, Policy, Return, Episode, and State. Common issues in ML are data quality and quantity, overfitting, underfitting, data bias, interpretability, computational resources, deployment challenges, security/privacy risks, and ethical concerns."
#     },
#     "DMandDMDWpdf__2025_06_07_15_49_49.pdf": {
#         "question": "What are the typical requirements of clustering in data mining?",
#         "answer": "Typical requirements include scalability, ability to deal with different types of attributes, discovery of clusters with arbitrary shape, minimal domain knowledge for input parameters, ability to handle noisy data, incremental clustering and input order insensitivity, high dimensionality handling, constraint-based clustering, and interpretability and usability."
#     },
#     "NIPS-2017-attention-is-all-you-need-Paper.pdf": {
#         "question": "What core innovation does the Transformer architecture introduce?",
#         "answer": "It replaces recurrence with self-attention mechanisms to model dependencies between input and output without relying on sequence order."
#     }
}

def citation_score(answer_obj, expected_sources=None):
    sources = answer_obj.get("sources", [])
    
    
    if sources:
        for source in sources:
            print(source)
            if("unknown" in source.lower()):
                return 0.2
            
        return 1.0

    # fallback: weak heuristic
    answer = answer_obj.get("answer", "")
    if ".pdf" in answer or "source:" in answer.lower():
        return 0.5  # partial score for at least mentioning something

    return 0




# --- Main Evaluation ---
def evaluate_model():
    results = []
    pdf_dir = './eval_pdfs'

    pdfs = [
        Path(os.path.join(pdf_dir, path))
        for path in os.listdir(pdf_dir)
        if path.lower().endswith('.pdf') and os.path.isfile(os.path.join(pdf_dir, path))
    ]

    if not pdfs:
        print("❌ No valid PDF/Text files found in 'eval_pdfs' directory.")
        return

    text = get_pdf_into_text(pdfs[:1])
    if not text.strip():
        print("❌ No text could be extracted from the PDF/Text files.")
        return

    chunks = create_chunks(text)
    if not chunks:
        print("❌ No text chunks were created.")
        return

    get_vector_store(chunks)

    for doc_name, qa in evaluation_data.items():
        question = qa["question"]
        expected_answer = qa["answer"]

        print(f"\n📄 Evaluating on: {doc_name}")
        print(f"❓ Question: {question}")

        try:
            response_obj = get_answer_from_chain(question)
            response = response_obj["answer"]

            # Similarity and completeness as usual
            score = similarity_score(response, expected_answer)
            complete_score = embedding_based_completeness(expected_answer, response)
            cite_score = citation_score(response_obj)



            print(f"✅ Expected: {expected_answer}")
            print(f"🤖 Response: {response}")
            print(f"📊 Similarity: {score:.4f}")
            print(f"📈 Completeness: {complete_score:.4f}")
            print(f"📚 Citation: {cite_score}")

            results.append({
                "document": doc_name,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": response,
                "similarity_score": round(score, 4),
                "completeness_score": round(complete_score, 4),
                "citation_score": cite_score
            })

        except Exception as e:
            logger.error(f"❌ Failed on {doc_name}: {str(e)}")
            results.append({
                "document": doc_name,
                "question": question,
                "expected_answer": expected_answer,
                "actual_answer": "Error: " + str(e),
                "similarity_score": 0.0
            })

    # Save to JSON and CSV
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    pd.DataFrame(results).to_csv("evaluation_results.csv", index=False)

    avg_score = sum(r["similarity_score"] for r in results) / len(results)
    print("\n🎯 Final Evaluation Result:")
    print(f"Average Similarity Score: {avg_score:.4f}")
    print(f"Average Completeness Score: {sum(r['completeness_score'] for r in results) / len(results):.4f}")
    print(f"Average Citation Score: {sum(r['citation_score'] for r in results) / len(results):.4f}")
    print(f"{'='*40}")

# --- Run ---
if __name__ == "__main__":
    evaluate_model()
