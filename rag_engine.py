import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    try:
        from langchain_classic.retrievers import EnsembleRetriever
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever
    except ImportError:
        from langchain_community.retrievers import BM25Retriever # already imported above
        from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
except ImportError:
    try:
        from langchain_classic.chains import create_retrieval_chain
        from langchain_classic.chains.combine_documents import create_stuff_documents_chain
    except ImportError:
        # Fallback to langchain_community or similar if possible, but these are core components
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Robust path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check both project root and web_app directory
ENV_LOCATIONS = [
    os.path.normpath(os.path.join(BASE_DIR, "..", ".env")),
    os.path.normpath(os.path.join(BASE_DIR, "..", "web_app", ".env"))
]
DB_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "code_travail_db"))

for env_path in ENV_LOCATIONS:
    if os.path.exists(env_path):
        print(f"Loading environment from: {env_path}")
        load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class RAGEngine:
    def __init__(self):
        if not GOOGLE_API_KEY:
            print("WARNING: GOOGLE_API_KEY not found in environment.")
            
        print(f"Initializing RAG Engine with DB at: {DB_DIR}")
        
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2-preview",
                task_type='RETRIEVAL_DOCUMENT',
                google_api_key=GOOGLE_API_KEY
            )
            
            self.vector_db = Chroma(
                collection_name='loi_maroc',
                embedding_function=self.embeddings,
                persist_directory=DB_DIR
            )
            
            self.llm = GoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                google_api_key=GOOGLE_API_KEY
            )
            
            self.chain = self._setup_chain()
            print("RAG Engine successfully initialized.")
        except Exception as e:
            print(f"FAILED to initialize RAG Engine: {e}")
            self.chain = None

    def _setup_chain(self):
        # 1. Vector Retriever
        vector_retriever = self.vector_db.as_retriever(search_kwargs={"k": 5})

        # 2. Multi-Query Retriever setup
        mq_retriever = MultiQueryRetriever.from_llm(
            retriever=vector_retriever, 
            llm=self.llm
        )

        # 3. Prompt Template
        system_prompt = (
            "Vous êtes un expert en droit du travail marocain. "
            "Utilisez les articles suivants pour répondre à l'utilisateur. "
            "Si vous ne trouvez pas la réponse dans le texte, dites que vous ne savez pas. "
            "Citez toujours le numéro de l'article pour justifier votre réponse.\n\n"
            "CONTEXTE LÉGAL:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}")
        ])

        # 4. Chains
        question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
        return create_retrieval_chain(mq_retriever, question_answer_chain)

    async def get_response(self, query: str):
        if not self.chain:
            return {"answer": "Erreur: Le moteur RAG n'est pas initialisé.", "context": []}
            
        try:
            response = await self.chain.ainvoke({"input": query})
            return {
                "answer": response["answer"],
                "context": [doc.page_content for doc in response["context"]]
            }
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                # Simple one-time retry after 2 seconds
                try:
                    import asyncio
                    await asyncio.sleep(2)
                    response = await self.chain.ainvoke({"input": query})
                    return {
                        "answer": response["answer"],
                        "context": [doc.page_content for doc in response["context"]]
                    }
                except:
                    return {
                        "answer": "Désolé, le service est temporairement saturé (Limite de quota atteinte). Veuillez réessayer dans une minute.",
                        "context": ["Status: Rate Limited"]
                    }
            
            # Log other errors but return a graceful message
            print(f"RAG Engine Error: {e}")
            safe_error = str(error_str)[:100]
            return {
                "answer": f"Une erreur technique est survenue: {safe_error}...",
                "context": ["Status: Error"]
            }

# Singleton instance
engine = RAGEngine()
