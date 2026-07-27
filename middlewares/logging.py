import time
import logging

from fastapi import Request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()
    # Read the body before calling the handler. Starlette caches it on the
    # request, so downstream handlers can still read it.
    body = await request.body()
    response = await call_next(request)
    response_time = time.perf_counter() - start_time
    body_text = body.decode("utf-8", errors="replace") if body else ""
    logger.info(
        f"{request.method} {request.url.path} {response.status_code} "
        f"{response_time:.3f}s body={body_text}"
    )
    return response
