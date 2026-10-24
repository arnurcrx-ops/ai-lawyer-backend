import os
import re
import logging
import datetime
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# RAG & ML imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer

# SQLAlchemy imports
from sqlalchemy import create_engine, Column, Integer, Text, DateTime
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ai-lawyer-backend")

load_dotenv()

app = FastAPI(title="Kazakhstan Legal Assistant API with RAG & PostgreSQL")

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key=os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-chat"

# --- 1. Database Setup (PostgreSQL with Graceful Degradation) ---
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ailawyer")

db_session_factory = None
Base = declarative_base()

class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(Text, nullable=False)
    bot_answer = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

try:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)
    db_session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    logger.info("Successfully connected to PostgreSQL database.")
except Exception as e:
    logger.warning(f"PostgreSQL is not available ({e}). Running without database persistence (Graceful Degradation).")

def save_chat_to_db(user_msg: str, bot_msg: str):
    if db_session_factory is None:
        return
    try:
        session = db_session_factory()
        history_item = ChatHistory(user_message=user_msg, bot_answer=bot_msg)
        session.add(history_item)
        session.commit()
        session.close()
    except Exception as e:
        logger.warning(f"Failed to save chat history to database: {e}")

# --- 2. RAG Setup (In-Memory Qdrant & Sentence Transformers) ---
logger.info("Loading embedding model (all-MiniLM-L6-v2)...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
vector_size = embedding_model.get_sentence_embedding_dimension()

qdrant_client = QdrantClient(location=":memory:")
COLLECTION_NAME = "labor_code_rk"

qdrant_client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
)

# Mock documents from RK legislation
mock_documents = [
    {
        "id": 1,
        "text": "Статья 113 ТК РК. Порядок и сроки выплаты заработной платы: Заработная плата выплачивается в национальной валюте Республики Казахстан не реже одного раза в месяц в первой декаде следующего месяца. При задержке выплаты зарплаты работодатель выплачивает задолженность и пеню за каждый день просрочки исходя из ставки рефинансирования Национального Банка РК."
    },
    {
        "id": 2,
        "text": "Статья 52 ТК РК. Основания расторжения трудового договора по инициативе работодателя: Трудовой договор по инициативе работодателя может быть расторгнут в случаях сокращения численности или штата работников, снижения объема производства, несоответствия работника занимаемой должности или повторного нарушения трудовой дисциплины."
    },
    {
        "id": 3,
        "text": "Статьи 49-50 ТК РК. Прекращение трудового договора: Трудовой договор может быть прекращен по соглашению сторон, истечению срока, расторжению по инициативе одной из сторон. При расторжении по сокращению штата работодатель обязан предупредить работника письменно не менее чем за 1 месяц."
    },
    {
        "id": 4,
        "text": "Закон РК 'О защите прав потребителей'. Возврат и обмен товаров: Потребитель вправе обменять или вернуть непродовольственный товар надлежащего качества в течение 14 дней, если он не был в употреблении. При обнаружении производственного брака потребитель вправе требовать возврата денег или замены товара."
    },
    {
        "id": 5,
        "text": "КоАП РК. Обжалование штрафов и ПДД: Предписание о наложении штрафа можно обжаловать в течение 10 суток со дня вручения. При уплате штрафа в течение 7 суток с момента получения (сокращенное производство) сумма штрафа уменьшается на 50%."
    }
]

# Populate Qdrant in-memory vector db
doc_texts = [doc["text"] for doc in mock_documents]
doc_embeddings = embedding_model.encode(doc_texts).tolist()

qdrant_client.upsert(
    collection_name=COLLECTION_NAME,
    points=[
        models.PointStruct(
            id=doc["id"],
            vector=embedding,
            payload={"text": doc["text"]}
        )
        for doc, embedding in zip(mock_documents, doc_embeddings)
    ]
)
logger.info("Loaded 5 mock legal documents into in-memory Qdrant RAG.")

def retrieve_relevant_context(query: str, top_k: int = 2) -> str:
    try:
        query_vector = embedding_model.encode(query).tolist()
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k
        )
        contexts = [hit.payload["text"] for hit in search_results if hit.payload]
        return "\n\n".join(contexts)
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}")
        return ""

# --- 3. Request / Response Models ---
class AskRequest(BaseModel):
    text: str

class AskResponse(BaseModel):
    answer: str

Req = AskRequest

# --- 4. System Prompt ---
BASE_SYSTEM_TEMPLATE = """Сен — Қазақстанға арналған AI-заңгерсің. Сен физикалық тұлғалар мен ЖК-ке ҚР заңдары бойынша көмектесесің.

ТІЛ ЕРЕЖЕСІ (КРИТИЧЕСКИ ВАЖНО):
1. Пайдаланушы қандай тілде жазса, сен ТОЛЫҚ СОЛ ТІЛДЕ жауап беруің керек (қазақша немесе орысша).
2. Заңдық терминдерді нақты қолдан.

РК ЗАҢНАМАСЫНАН АЛЫНҒАН ДЕРЕКТЕР (ҚОЛДАНУ МІНДЕТТІ):
{context}

НҰСҚАУ: Жоғарыдағы құжаттар мен ҚР заңдарына сүйене отырып жауап бер.

ЖАУАП ҚҰРЫЛЫМЫ:
Тек таза мәтін, ешқандай markdown (#, *, ```) белгілерінсіз.
Қорытынды (Вывод): не істеу керек.
Жоспар (План):
1. ...
2. ...
Үлгі (Шаблон): ...
ҚР НҚА (НПА РК): ...
Мерзімдер (Сроки): ...
Ескерту (Дисклеймер): ⚖️ Бұл ақпараттық анықтама, заңдық кеңес емес."""

def clean_answer(text: str) -> str:
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = text.replace('**', '').replace('*', '')
    text = text.replace('```', '').replace('`', '')
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = text.replace('__', '').replace('_', '')
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

@app.post("/ask", response_model=AskResponse)
def ask(r: Req):
    try:
        # Retrieve RAG context
        context_text = retrieve_relevant_context(r.text, top_k=2)
        system_prompt = BASE_SYSTEM_TEMPLATE.format(context=context_text if context_text else "Жалпы ҚР заңнамасы")

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/AI-Lawyer-KZ",
            "X-Title": "AI Lawyer KZ"
        }

        payload = {
            "model": DEFAULT_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": r.text}
            ],
            "temperature": 0.3
        }

        response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)

        if response.status_code != 200:
            fallback = generate_fallback_response(r.text, f"OpenRouter status code: {response.status_code}")
            cleaned = clean_answer(fallback)
            save_chat_to_db(r.text, cleaned)
            return AskResponse(answer=cleaned)

        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            fallback = generate_fallback_response(r.text, "No choices returned from OpenRouter")
            cleaned = clean_answer(fallback)
            save_chat_to_db(r.text, cleaned)
            return AskResponse(answer=cleaned)

        raw = choices[0].get("message", {}).get("content", "").strip()
        if not raw:
            fallback = generate_fallback_response(r.text, "Empty content from OpenRouter")
            cleaned = clean_answer(fallback)
            save_chat_to_db(r.text, cleaned)
            return AskResponse(answer=cleaned)

        cleaned = clean_answer(raw)
        save_chat_to_db(r.text, cleaned)
        return AskResponse(answer=cleaned)

    except Exception as e:
        fallback = generate_fallback_response(r.text, str(e))
        cleaned = clean_answer(fallback)
        save_chat_to_db(r.text, cleaned)
        return AskResponse(answer=cleaned)

def generate_fallback_response(query: str, error: str) -> str:
    lower_query = query.lower()
    if any(k in lower_query for k in ["зарплат", "деньги", "долг", "зп", "жалақы"]):
        legal_advice = (
            "Қорытынды: Еңбекақы қарызын өндіріп алу үшін жұмыс берушіге және сотқа талап қойыңыз.\n\n"
            "Жоспар:\n"
            "1. Жұмыс берушіге жазбаша талап арыз беріңіз.\n"
            "2. Егер төлемесе, Еңбек инспекциясына шағым түсіріңіз.\n"
            "3. Сотқа талап арыз беріңіз.\n\n"
            "Үлгі (Шаблон):\n"
            "Жалақыны төлеу туралы талап арыз.\n\n"
            "ҚР НҚА (НПА РК): ҚР Еңбек кодексінің 113-бабы.\n"
            "Мерзімдер: Сотқа жүгіну мерзімі — 3 ай.\n"
            "Ескерту: ⚖️ Бұл ақпараттық анықтама, заңдық кеңес емес."
        )
    else:
        legal_advice = (
            f"Қорытынды: Құқықтық сұрағыңызды нақтылап, мамандарға жүгініңіз.\n\n"
            f"Жоспар:\n"
            f"1. Барлық құжаттарды жинаңыз.\n"
            f"2. ҚР заңнамасын қараңыз.\n"
            f"3. egov.kz немесе 1414 нөміріне хабарласыңыз.\n\n"
            f"Үлгі (Шаблон):\n"
            f"Мемлекеттік органдарға сұрау салу.\n\n"
            f"ҚР НҚА (НПА РК): Қазақстан Республикасының заңнамасы.\n"
            f"Мерзімдер: Талап қою мерзімі — 3 жыл.\n"
            f"Ескерту: ⚖️ Бұл ақпараттық анықтама, заңдық кеңес емес."
        )
    return legal_advice

@app.get("/")
def root():
    return {"message": "Kazakhstan Legal Assistant API with RAG & PostgreSQL is running"}
