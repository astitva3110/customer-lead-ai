import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.error_handlers import register_error_handlers
from app.routers import auth, channels, chat, customer_service, health, knowledge, leads, users
from app.services.diagnostics.langsmith_tracing import configure_langsmith

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="earKART Chatbot",
    description="Document-first knowledge base chatbot with vector retrieval.",
    version="0.3.0",
)

_cors_origins = [item.strip() for item in (settings.cors_allow_origins or "").split(",") if item.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

register_error_handlers(app)
configure_langsmith()

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(leads.router)
app.include_router(customer_service.router)
app.include_router(chat.router)
app.include_router(channels.router)
app.include_router(knowledge.router)
