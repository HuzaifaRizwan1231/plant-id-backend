from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.config import FRONTEND_URL

# Initialize FastAPI app
app = FastAPI()

origins= [
    FRONTEND_URL,
]

# CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Routes
# app.include_router(chat_routes.router, prefix="/api/chat")


    

