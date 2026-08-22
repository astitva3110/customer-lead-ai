from fastapi import FastAPI

from app.routers import auth, chat, health, knowledge, leads, users

app = FastAPI(
    title="earKART Chatbot",
    description="Document-first knowledge base chatbot with vector retrieval.",
    version="0.3.0",
)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(leads.router)
app.include_router(chat.router)
app.include_router(knowledge.router)
