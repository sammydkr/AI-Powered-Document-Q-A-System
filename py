# requirements.txt
# azure-identity==1.15.0
# azure-search-documents==11.4.0
# azure-storage-blob==12.19.0
# openai==1.3.0
# python-dotenv==1.0.0
# pypdf==3.17.0
# streamlit==1.28.0

import os
import json
import tempfile
from typing import List, Dict
import PyPDF2
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchFieldDataType,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField,
    SemanticSettings
)
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

class DocumentQASystem:
    def __init__(self):
        self.azure_openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2023-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        self.search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=os.getenv("AZURE_SEARCH_INDEX_NAME"),
            credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
        )
        
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        self.chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        text = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.read())
            tmp_file.flush()
            
            with open(tmp_file.name, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        
        os.unlink(tmp_file.name)
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000) -> List[str]:
        """Split text into manageable chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word) + 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings from Azure OpenAI"""
        response = self.azure_openai_client.embeddings.create(
            input=text,
            model=self.embedding_deployment
        )
        return response.data[0].embedding
    
    def index_document(self, document_text: str, document_name: str):
        """Index document chunks in Azure Cognitive Search"""
        chunks = self.chunk_text(document_text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            embedding = self.get_embeddings(chunk)
            doc = {
                "id": f"{document_name}_{i}",
                "content": chunk,
                "document_name": document_name,
                "embedding": embedding
            }
            documents.append(doc)
        
        self.search_client.upload_documents(documents=documents)
    
    def search_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar document chunks"""
        query_embedding = self.get_embeddings(query)
        
        results = self.search_client.search(
            search_text=query,
            vector=query_embedding,
            top=top_k,
            vector_fields="embedding"
        )
        
        return [{"content": result["content"], "score": result["@search.score"]} 
                for result in results]
    
    def answer_question(self, question: str, context_chunks: List[Dict]) -> str:
        """Generate answer using context from documents"""
        context = "\n\n".join([chunk["content"] for chunk in context_chunks])
        
        prompt = f"""Based on the following context, please answer the question. 
        If the answer cannot be found in the context, say "I cannot find the answer in the provided documents."

        Context:
        {context}

        Question: {question}

        Answer:"""
        
        response = self.azure_openai_client.chat.completions.create(
            model=self.chat_deployment,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        return response.choices[0].message.content

# Streamlit UI
import streamlit as st

def main():
    st.title("📚 AI Document Q&A System")
    st.write("Upload a PDF and ask questions about its content!")
    
    # Initialize system
    if 'qa_system' not in st.session_state:
        st.session_state.qa_system = DocumentQASystem()
    
    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process Document"):
            with st.spinner("Processing document..."):
                text = st.session_state.qa_system.extract_text_from_pdf(uploaded_file)
                st.session_state.qa_system.index_document(text, uploaded_file.name)
                st.success("Document processed and indexed!")
    
    # Question answering
    question = st.text_input("Ask a question about the document:")
    
    if question and uploaded_file is not None:
        with st.spinner("Searching for answers..."):
            similar_chunks = st.session_state.qa_system.search_similar_chunks(question)
            answer = st.session_state.qa_system.answer_question(question, similar_chunks)
            
            st.subheader("Answer:")
            st.write(answer)
            
            st.subheader("Relevant Context:")
            for i, chunk in enumerate(similar_chunks, 1):
                st.write(f"**Chunk {i} (Score: {chunk['score']:.2f}):**")
                st.write(chunk["content"][:200] + "...")

if __name__ == "__main__":
    main()
