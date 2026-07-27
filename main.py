from fastapi import FastAPI

from controllers.learning import router as learning_router
from controllers.roleplay import router as roleplay_router
from middlewares.auth import basic_auth_middleware
from middlewares.logging import log_requests

app = FastAPI()

app.middleware("http")(log_requests)
app.middleware("http")(basic_auth_middleware)
app.include_router(roleplay_router)
app.include_router(learning_router)


@app.get("/ping")
def ping():
    return {"ping": "pong"}



