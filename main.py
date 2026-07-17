from fastapi import FastAPI

from controllers.roleplay import router as roleplay_router
from middlewares.auth import basic_auth_middleware

app = FastAPI()

app.middleware("http")(basic_auth_middleware)
app.include_router(roleplay_router)


@app.get("/ping")
def ping():
    return {"ping": "pong"}



