from fastapi import FastAPI

from controllers.roleplay import router as roleplay_router

app = FastAPI()

app.include_router(roleplay_router)


@app.get("/ping")
def ping():
    return {"ping": "pong"}



