# pyright: reportGeneralTypeIssues=false
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Optional

import pendulum
import uvicorn
from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
import json
import urllib.request



top_url = 'https://hacker-news.firebaseio.com/v0/topstories.json?print=pretty'
base_url = 'https://hacker-news.firebaseio.com/v0/item/{}.json?print=pretty'

def get_top_story_ids():
    response = urllib.request.urlopen(top_url)
    top_story_ids = json.loads(response.read().decode('utf-8'))
    return top_story_ids

def get_top_stories(top_story_ids):
    stories = []
    for story_id in top_story_ids:
        response = urllib.request.urlopen(base_url.format(story_id))
        story_data = json.loads(response.read().decode('utf-8'))
        stories.append(story_data)
    return stories





@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:

    mclient = Client("127.0.0.1", 11211)
    FastAPICache.init(MemcachedBackend(mclient), prefix="fastapi-cache")
    yield



@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    FastAPICache.init(InMemoryBackend())
    yield


app = FastAPI(lifespan=lifespan)

news = {'First news key here':'news description and other stuff here'}

count =5

ret=0

@cache(namespace="test", expire=5)
async def get_news(count:int = 5):
    global news

    top_story_ids = get_top_story_ids()
    news = get_top_stories(top_story_ids[:count])
    return news


@app.get("/")
@cache(namespace="test", expire=10)
async def index():
    global news
    return news

@app.get("/news/")
@cache(namespace="test", expire=10)
async def index(top:int=5):

    return {"ret":await get_news(top)}

@app.get("/clear")
async def clear():
    return await FastAPICache.clear(namespace="test")


@app.get("/date")
@cache(namespace="test", expire=10)
async def get_date():
    return pendulum.today()


@app.get("/datetime")
@cache(namespace="test", expire=2)
async def get_datetime(request: Request, response: Response):
    return {"now": pendulum.now()}


@cache(namespace="test")
async def func_kwargs(*unused_args, **kwargs):
    return kwargs


@app.get("/kwargs")
async def get_kwargs(name: str):
    return await func_kwargs(name, name=name)


@app.get("/sync-me")
@cache(namespace="test") # pyright: ignore[reportArgumentType]
def sync_me():
    # as per the fastapi docs, this sync function is wrapped in a thread,
    # thereby converted to async. fastapi-cache does the same.
    return 42


@app.get("/cache_response_obj")
@cache(namespace="test", expire=5)
async def cache_response_obj():
    return JSONResponse({"a": 1})


class SomeClass:
    def __init__(self, value):
        self.value = value

    async def handler_method(self):
        return self.value


# register an instance method as a handler
instance = SomeClass(17)
app.get("/method")(cache(namespace="test")(instance.handler_method))


# cache a Pydantic model instance; the return type annotation is required in this case
class Item(BaseModel):
    name: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None


@app.get("/pydantic_instance")
@cache(namespace="test", expire=5)
async def pydantic_instance() -> Item:
    return Item(name="Something", description="An instance of a Pydantic model", price=10.5)


put_ret = 0





@app.get("/namespaced_injection")
@cache(namespace="test", expire=5, injected_dependency_namespace="monty_python") # pyright: ignore[reportArgumentType]
def namespaced_injection(
    __fastapi_cache_request: int = 42, __fastapi_cache_response: int = 17
) -> Dict[str, int]:
    return {
        "__fastapi_cache_request": __fastapi_cache_request,
        "__fastapi_cache_response": __fastapi_cache_response,
    }




if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
