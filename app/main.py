import logging

from fastapi import FastAPI

from app.error_handlers import register_error_handlers
from app.routers import auth, chat, customer_service, health, knowledge, leads, users

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(
    title="earKART Chatbot",
    description="Document-first knowledge base chatbot with vector retrieval.",
    version="0.3.0",
)

register_error_handlers(app)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(leads.router)
app.include_router(customer_service.router)
app.include_router(chat.router)
app.include_router(knowledge.router)
