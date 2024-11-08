
from routes.url_api import router as url_router

from fastapi import FastAPI

app = FastAPI()
app.include_router(url_router)
