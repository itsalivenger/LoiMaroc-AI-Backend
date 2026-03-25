<div align="center">

# ⚖️ LoiMaroc AI — Backend

<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
<img src="https://img.shields.io/badge/LangChain-RAG-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white" />
<img src="https://img.shields.io/badge/Google_Gemini_2.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white" />
<img src="https://img.shields.io/badge/MongoDB-Motor-47A248?style=for-the-badge&logo=mongodb&logoColor=white" />
<img src="https://img.shields.io/badge/ChromaDB-Vector_Store-FFA500?style=for-the-badge" />

<br/><br/>

> 🧠 **Le cerveau de LoiMaroc AI.**
> Un backend RAG haute performance dédié au droit marocain du travail.

</div>

---

## 🌟 Architecture Générale

Ce backend est le moteur de **LoiMaroc AI**. Il expose une API REST FastAPI qui :

1. **Répond aux questions juridiques** en interrogeant une base vectorielle ChromaDB enrichie du Code du Travail marocain
2. **Gère l'authentification** des utilisateurs (inscription, vérification email OTP, connexion)
3. **Persiste les conversations** dans MongoDB pour chaque utilisateur
4. **Fournit un panneau admin** pour gérer utilisateurs, sessions et configuration globale
5. **Envoie des emails transactionnels** (vérification, contact) via SMTP

---

## 🧰 Stack Technique

| Technologie | Rôle |
|---|---|
| **FastAPI** | Framework API REST asynchrone — routes, middlewares, validation |
| **Uvicorn** | Serveur ASGI haute performance pour Python |
| **Motor** | Driver MongoDB asynchrone (non-bloquant) |
| **Pydantic v2** | Validation et sérialisation des données |
| **LangChain** | Orchestration du pipeline RAG (chaînes, retrievers, prompts) |
| **Google Gemini 2.5 Flash** | LLM pour la génération de réponses juridiques |
| **Google Gemini Embeddings** | Modèle d'embeddings pour la recherche sémantique |
| **ChromaDB** | Base de données vectorielle persistante (vector store) |
| **BM25Retriever** | Recherche lexicale (complément au retriever vectoriel) |
| **EnsembleRetriever** | Fusionne les résultats BM25 + vectoriel pour de meilleurs rappels |
| **MultiQueryRetriever** | Génère plusieurs variantes de la question pour maximiser la pertinence |
| **python-dotenv** | Chargement des variables d'environnement |
| **smtplib / email.mime** | Envoi d'emails transactionnels (vérification, contact) |

---

## 🧩 Fichiers Principaux

```
backend/
├── main.py          # Application FastAPI : routes API, auth, admin, contact, sessions
├── rag_engine.py    # Moteur RAG : ChromaDB + Gemini + LangChain pipeline
├── requirements.txt # Dépendances Python
└── .gitignore       # Exclusions Git (venv, .env, cache, FAISS...)
```

### `main.py` — API REST
Toutes les routes HTTP de l'application :

| Méthode | Route | Description |
|---|---|---|
| `GET` | `/` | Health check de base |
| `GET` | `/api/health` | Status détaillé (MongoDB + RAG Engine) |
| `POST` | `/api/chat` | 💬 Requête au moteur RAG |
| `POST` | `/api/auth/register` | Inscription + envoi OTP email |
| `POST` | `/api/auth/verify` | Validation du code OTP |
| `POST` | `/api/auth/login` | Connexion utilisateur |
| `GET` | `/api/sessions/user/{email}` | Sessions d'un utilisateur |
| `POST` | `/api/sessions` | Sauvegarde/mise à jour d'une session |
| `POST` | `/api/admin/login` | Connexion admin |
| `GET` | `/api/admin/config` | Lecture de la configuration globale |
| `POST` | `/api/admin/config` | Mise à jour de la configuration |
| `GET` | `/api/admin/users` | Liste tous les utilisateurs |
| `DELETE` | `/api/admin/users/{email}` | Supprime un utilisateur |
| `PATCH` | `/api/admin/users/{email}/verify` | Bascule la vérification d'un compte |
| `GET` | `/api/admin/sessions` | Toutes les conversations |
| `DELETE` | `/api/admin/sessions/{id}` | Supprime une conversation |
| `POST` | `/api/contact` | Envoi d'un message de contact par email |

### `rag_engine.py` — Moteur RAG
Pipeline de recherche et génération :

```
Question ──► MultiQueryRetriever ──► EnsembleRetriever (BM25 + ChromaDB)
                                              │
                                      Documents Juridiques
                                              │
                              Gemini 2.5 Flash + Prompt Expert
                                              │
                                        Réponse sourcée ✅
```

---

## 🚀 Lancer le Backend

### Prérequis

- **Python 3.10+**
- **MongoDB** local ou Atlas
- **Clé API Google** (Gemini)
- La base vectorielle ChromaDB déjà indexée dans `../code_travail_db/`

### Installation

```bash
# 1. Créer et activer un environnement virtuel
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Lancer le serveur de développement
uvicorn main:app --reload --port 8000
```

L'API sera disponible sur 👉 **http://localhost:8000**

La documentation interactive Swagger est accessible sur **http://localhost:8000/docs** 📚

---

## 🗄️ Collections MongoDB

| Collection | Contenu |
|---|---|
| `users` | Comptes utilisateurs vérifiés |
| `pending_registrations` | Inscriptions en attente de vérification OTP |
| `sessions` | Historique des conversations par email |
| `admins` | Comptes administrateurs |
| `settings` | Configuration globale de l'application |

---

## 🔐 Sécurité

- Les mots de passe sont actuellement stockés en clair — **à hasher avec `bcrypt` en production**
- Les emails de vérification expirent après **15 minutes**
- Les codes OTP sont des nombres à **6 chiffres** générés aléatoirement
- L'envoi des emails utilise `formataddr` pour afficher **"LoiMaroc AI"** comme expéditeur

---

<div align="center">

**Made with 🐍 & ☕ · LoiMaroc AI Backend © 2025**

*La loi marocaine, accessible à tous — propulsée par Gemini.*

</div>
