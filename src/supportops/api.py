"""Loopback-only demo API. No accounts, uploads, personal data or financial actions."""
import logging
import threading
import time
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse

from .agent import run_agent, run_rag
from .classifier import predict
from .llm import ModelUnavailable
from .paths import WEB
from .retrieval import Retriever

app = FastAPI(title='SupportOps AI', version='0.1.0')
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['localhost', '127.0.0.1', 'testserver'])
log = logging.getLogger('supportops')
hits = defaultdict(deque)
hits_lock = threading.Lock()
model_lock = threading.Lock()


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    text: str = Field(min_length=3, max_length=2000)


@app.middleware('http')
async def guard(request: Request, call_next):
    origin = request.headers.get('origin')
    if request.method == 'POST' and origin and origin not in {'http://localhost:8765','http://127.0.0.1:8765'}:
        return JSONResponse({'detail':'Origem não autorizada.'}, status_code=403)
    if request.url.path.startswith('/api/'):
        key = (request.client.host if request.client else 'unknown', request.url.path)
        with hits_lock:
            now = time.monotonic()
            q = hits[key]
            while q and q[0] < now - 60:
                q.popleft()
            limit = 5 if request.url.path in {'/api/agent','/api/rag'} else 40
            if len(q) >= limit:
                log.warning('rate_limit path=%s', request.url.path)
                return JSONResponse({'detail':'Limite temporário atingido. Tente em um minuto.'},status_code=429)
            q.append(now)
    response = await call_next(request)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    return response


@app.get('/health')
def health():
    return {'status':'ok','financial_actions':False,'storage':'none'}


@app.post('/api/classify')
def classify(body: Question):
    return predict(body.text)


@app.post('/api/search')
def search(body: Question):
    return {'documents':Retriever().search(body.text),'retrieval':'lexical'}


def invoke_model(fn, text):
    if not model_lock.acquire(blocking=False):
        raise HTTPException(status_code=429,detail='O modelo está ocupado. Aguarde a execução atual.')
    try:
        return fn(text, semantic=True)
    except (ModelUnavailable, ValueError):
        raise HTTPException(status_code=503,detail='Modelo ou índice local indisponível. Consulte as instruções de execução.')
    finally:
        model_lock.release()


@app.post('/api/agent')
def agent(body: Question):
    return invoke_model(run_agent, body.text)


@app.post('/api/rag')
def rag(body: Question):
    return invoke_model(run_rag, body.text)


@app.get('/')
def home():
    return FileResponse(WEB / 'index.html')


app.mount('/', StaticFiles(directory=str(WEB)), name='web')
