import google.generativeai as genai
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from core.splitter import split_text
from core.vectorstore import save_to_faiss, load_faiss, LocalEmbeddingFunction
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI 
GEMINI_API_KEY =os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
sentence_transformer_model = SentenceTransformer("all-MiniLM-L6-v2")
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    google_api_key=GEMINI_API_KEY 
)
def get_response(prompt: str):

    embedding_function = LocalEmbeddingFunction(model=sentence_transformer_model)
    db = load_faiss(embedding_model=embedding_function, path="faiss_index") 
    retriever = db.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True 
    )
    response = qa_chain.invoke({"query": prompt}) 

        
    answer = response.get("result", "No answer found.")
    return answer

if __name__ == "__main__":
    prompt_question = "What is the main topic of the document?"
    response = get_response(prompt_question)
    print(f"\nQuestion: {prompt_question}")
    print(f"Answer: {response}")