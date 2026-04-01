import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
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
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Robust path handling - important for Vercel vs Local
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Check PROJECT ROOT (..) and WEB_APP directory (sibling of PROJECT ROOT)
ENV_LOCATIONS = [
    os.path.join(BASE_DIR, ".env"),
    os.path.join(BASE_DIR, "..", ".env"),
    os.path.join(BASE_DIR, "..", "..", "web_app", ".env")
]
# Try to find the DB folder in multiple locations
DB_POSSIBILITIES = [
    os.path.join(BASE_DIR, "code_travail_db"),
    os.path.join(BASE_DIR, "..", "code_travail_db"),
    "/tmp/code_travail_db"
]

DB_DIR = DB_POSSIBILITIES[0]
for p in DB_POSSIBILITIES:
    if os.path.exists(p) and os.path.isdir(p):
        DB_DIR = p
        break

for env_path in ENV_LOCATIONS:
    if os.path.exists(env_path):
        print(f"Loading environment from: {env_path}")
        load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class RAGEngine:
    def __init__(self):
        self.chain = None
        self.llm = None
        self.embeddings = None
        self.vector_db = None
        self.error_msg = None
        
        print(f"Initializing RAG Engine with DB at: {DB_DIR}")
        
        try:
            if not GOOGLE_API_KEY:
                raise ValueError("Missing GOOGLE_API_KEY")

            # 1. Initialize LLM First - Most likely to succeed and useful for fallback
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-3-flash-preview",
                temperature=0,
                google_api_key=GOOGLE_API_KEY
            )

            # 2. Then try to load legal database
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2-preview",
                task_type='RETRIEVAL_DOCUMENT',
                google_api_key=GOOGLE_API_KEY
            )
            
            # Check if DB_DIR exists before loading
            if not os.path.exists(DB_DIR):
                print(f"CRITICAL: DB_DIR {DB_DIR} DOES NOT EXIST.")
                # We'll still try but it will likely fail or create empty
            
            self.vector_db = Chroma(
                collection_name='loi_maroc',
                embedding_function=self.embeddings,
                persist_directory=DB_DIR
            )
            
            # 3. Setup Chain
            self.chain = self._setup_chain()
            print("RAG Engine successfully initialized.")
        except Exception as e:
            print(f"FAILED to initialize RAG Engine: {e}")
            self.error_msg = str(e)
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
            "Tu es **Omar**, un assistant juridique intelligent et bienveillant, "
            "spécialisé exclusivement dans le **droit du travail marocain** (Code du Travail, Dahirs, décrets d'application).\n\n"

            "## DIRECTIVES CRUCIALES\n"
            "1. **CONCISION ABSOLUE** : Réponds de manière directe et précise. Évite les longs paragraphes inutiles.\n"
            "2. **PAS DE RÉPÉTITION D'INTRODUCTION** : Si la conversation a déjà commencé, NE DIS PAS 'Bonjour je suis Omar'. Ne te présente qu'une seule fois au début du chat.\n"
            "3. **STRICTEMENT SUR LE CONTEXTE** : Ne réponds qu'aux questions liées au droit du travail marocain. Si une question est hors-sujet, refuse poliment.\n\n"

            "## TON RÔLE\n"
            "Aider citoyens, salariés et employeurs à comprendre leurs droits et obligations "
            "de manière claire, précise et accessible.\n\n"

            "## RÈGLES DE RÉPONSE\n"
            "1. **Toujours répondre en français**, avec un ton professionnel.\n"
            "2. **Structurer les réponses** : utilise des listes, des étapes numérotées ou des paragraphes courts.\n"
            "3. **Citer les articles** : mentionne toujours les numéros d'articles (ex : « Selon l'article 34 du Code du Travail... »).\n"
            "4. **Ne jamais inventer** d'articles ou de lois.\n\n"

            "## CAS PARTICULIERS\n"
            "- Si aucun article pertinent n'est trouvé : dis-le clairement et recommande un professionnel.\n"
            "- Si la question est ambiguë : demande une précision.\n\n"

            "## ARTICLES JURIDIQUES RÉCUPÉRÉS\n"
            "{context}\n\n"
            "Utilise ces articles pour construire ta réponse. S'ils sont vides ou non pertinents, "
            "applique la règle 'Aucun article pertinent' ci-dessus."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}")
        ])

        # 4. Custom context formatter that includes metadata
        def format_docs(docs):
            parts = []
            for doc in docs:
                meta = doc.metadata or {}
                article = meta.get("article", meta.get("article_number", ""))
                source = meta.get("source", meta.get("title", ""))
                header = "---"
                if article:
                    header = f"[Article {article}]"
                    if source:
                        header += f" — {source}"
                elif source:
                    header = f"[{source}]"
                parts.append(f"{header}\n{doc.page_content}")
            return "\n\n".join(parts)

        question_answer_chain = create_stuff_documents_chain(
            self.llm, prompt,
            document_variable_name="context",
            document_separator="\n\n",
        )
        # Patch the document formatter on the chain if possible
        try:
            question_answer_chain.document_prompt.template = "{page_content}"
        except Exception:
            pass

        self._format_docs = format_docs
        return create_retrieval_chain(mq_retriever, question_answer_chain)

    async def get_response(self, query: str, history: List[Dict[str, str]] = None):
        if not self.chain:
            # Try to re-init if failed before? Or fall back to Gemini directly
            print("RAG Engine: Chain not ready. Trying fallback.")
            return await self._gemini_fallback(query, history)
            
        try:
            # Convert history to LangChain messages if provided
            chat_history = []
            if history:
                for msg in history:
                    if msg["role"] == "user":
                        chat_history.append(("human", msg["content"]))
                    else:
                        chat_history.append(("assistant", msg["content"]))

            response = await self.chain.ainvoke({
                "input": query,
                "chat_history": chat_history
            })
            context_docs = response.get("context", [])
            context_text = " ".join(doc.page_content for doc in context_docs).strip()

            # If RAG found nothing meaningful, fall back to direct Gemini call
            if len(context_text) < 100:
                return await self._gemini_fallback(query)

            # Extract source references from metadata
            sources = []
            for doc in context_docs:
                meta = doc.metadata or {}
                article = meta.get("article", meta.get("article_number", ""))
                source = meta.get("source", meta.get("title", ""))
                ref = ""
                if article:
                    ref = f"Article {article}"
                    if source:
                        ref += f" — {source}"
                elif source:
                    ref = source
                elif doc.page_content:
                    # Fallback: first 80 chars of the chunk as reference
                    ref = doc.page_content[:80].strip() + "..."
                if ref and ref not in sources:
                    sources.append(ref)

            return {
                "answer": response["answer"],
                "context": sources if sources else [doc.page_content[:80] + "..." for doc in context_docs]
            }
        except Exception as e:
            # Log other errors but return a graceful message
            print(f"RAG Engine Error: {e}")
            safe_error = str(e)[:100]
            return {
                "answer": f"Une erreur technique est survenue: {safe_error}...",
                "context": ["Status: Error"]
            }

    async def _gemini_fallback(self, query: str, history: List[Dict[str, str]] = None) -> dict:
        """Direct Gemini call when RAG context is insufficient."""
        if not self.llm:
            return {
                "answer": "Désolé, je ne peux pas répondre car mon cerveau (LLM) n'est pas initialisé.",
                "context": []
            }

        # Build context from history
        history_context = ""
        if history:
            history_context = "Historique de la conversation :\n"
            for msg in history[-5:]: # Last 5 messages for brevity
                role = "Utilisateur" if msg["role"] == "user" else "Omar"
                history_context += f"{role}: {msg['content']}\n"
            history_context += "\n"

        fallback_prompt = (
            f"Tu es Omar, un assistant juridique spécialisé en droit du travail marocain.\n"
            f"{history_context}"
            f"DIRECTIVES :\n"
            f"- RÉPONSE CONCISE : Pas de longs discours, va droit au but.\n"
            f"- PAS DE RÉPÉTITION : Si l'utilisateur a déjà dit bonjour ou que la conversation est en cours, ne te représente pas.\n"
            f"- Ma base de données d'articles n'a pas trouvé de texte spécifique, réponds avec ta connaissance générale.\n\n"
            f"Question : {query}"
        )
        try:
            response = await self.llm.ainvoke(fallback_prompt)
            
            # Properly extract text content from AIMessage
            if isinstance(response.content, str):
                answer_text = response.content
            elif isinstance(response.content, list):
                # Join all text parts if it's a list (common in newer Gemini models)
                answer_text = "".join([part.get("text", "") for part in response.content if isinstance(part, dict) and part.get("type") == "text"])
            else:
                answer_text = str(response.content)

            return {
                "answer": answer_text,
                "context": []
            }
        except Exception as e:
            print(f"Gemini fallback error: {e}")
            error_details = str(e)[:100]
            return {
                "answer": f"Désolé, je rencontre une difficulté technique pour accéder à mes connaissances (Erreur: {error_details}). Veuillez vérifier votre connexion ou réessayer plus tard.",
                "context": []
            }

# Singleton instance
engine = RAGEngine()
