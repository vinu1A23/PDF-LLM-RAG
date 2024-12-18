# pyright: reportGeneralTypeIssues=false
from contextlib import asynccontextmanager
from typing import AsyncIterator #, Dict, Optional

#import pendulum
import uvicorn
from T13_subt1_rag_backend import *
from fastapi import FastAPI


"""
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

"""

def extract_text_from_pdf(pdf_content):
    docs = pdf_content
    #app.logger.info("len of docs is" + str(len(docs)))
    #logger.info("len of docs is" + str(len(docs)))
    all_data=""
    for i in range(len(docs)):
        all_data += docs[i].page_content[0:]
    #app.logger.info(f"Printing a bit of all_data {all_data[:100]}")
    #logger.info(f"Printing a bit of all_data {all_data[:100]}")
    return all_data





app = FastAPI()

news = {'First news key here':'news description and other stuff here'}

count =5

ret=0

async def get_news(url,name):
    

    pdf_await = await download_pdf(url,name)
    
        
    return news


@app.get("/")
async def index():
    global news
    return news

@app.get("/news/")
async def index(url,name):

    return {"ret":await get_news(url,name)}
"""
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



"""
if __name__ == "__main__":
    uvicorn.run("T13_subt1_main:app", reload=True)
