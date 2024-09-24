
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
"""from transformers import AutoTokenizer,  AutoModelForCausalLM """
from langchain_community.document_loaders import PyPDFLoader
import os
import urllib.request
import shutil
import logging
import aiohttp
import aiofiles
import asyncio

cache_directory = os.getenv("cache_directory")



device = 'cpu'

logger = logging.getLogger(__name__)
logging.basicConfig(filename='Test11_backend.log', encoding='utf-8', level=logging.DEBUG)


async def download_pdf(url,name):
    if url is None or url == "":
        file_url = "https://arxiv.org/pdf/1906.08172"
    else:
        file_url = url
    if name is None or name =="":
        file_name = "Mediapipe.pdf"
    else:
        file_name = name

    logger.info(f"file_name is ,{file_name}")
    async with aiohttp.ClientSession() as session:
        async with session.get(file_url) as response:
            async with aiofiles.open(file_name, 'wb') as out_file:
                content = await response.read()
                return await out_file.write(content)


async def load_pdf(name):

    if name is None or name == "":
        file_path = "Mediapipe.pdf"
    else :
        file_path = name
    logger.info(f"file path given as  {file_path}")
    loader = await  asyncio.to_thread(PyPDFLoader,file_path)
    return loader.load()


async def split_doc(content):

    # Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
    # It splits text into chunks of 1000 characters each with a 150-character overlap.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    if content is None or content == "":
        docs = PyPDFLoader("Mediapipe.pdf").load()
    else:
        docs = content

    # 'data' holds the text you want to split, split the text into documents
    # using the text splitter.
    document_splitted = text_splitter.split_documents(docs)
    logger.info("\n***** Logging the first doc split**** \n" ,document_splitted[0])

    return document_splitted


def load_embedding(
    modelPath="Alibaba-NLP/gte-multilingual-base",
    model_kwargs={'device':'cpu', 'trust_remote_code':'True'},
    encode_kwargs={'normalize_embeddings': False},
    cache_directory=cache_directory
     ):
    # Initialize an instance of HuggingFaceEmbeddings with the specified parameters

    embeddings = HuggingFaceEmbeddings(
        model_name=modelPath,  # Provide the pre-trained model's path
        model_kwargs=model_kwargs,  # Pass the model configuration options
        encode_kwargs=encode_kwargs,  # Pass the encoding options
        cache_folder=cache_directory
    )

    text = "This is a test document."
    query_result = embeddings.embed_query(text)
    logger.info(" Printing result of query result "+str(query_result[:3]))
    return embeddings


def vector_database_setup(document_splitted, embeddings):

    db = FAISS.from_documents(document_splitted, embeddings)
    return db


def load_model(
    model_name="Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="cpu",
    cache_dir=cache_directory,
    max_length=1024
    ):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        cache_dir=cache_dir
        )
    # Load the tokenizer associated with the specified model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, padding=True,
        truncation=True,
        max_length=max_length,
        cache_dir=cache_directory
        )
    return model, tokenizer


def generate_context(db, query):
    if query is None or query == "":
        question = "What is architecture used?"
    else:
        question = query
    searchDocs = db.similarity_search(question)
    logger.info(" *****, Q was , " + question+ "***answer is within doc split ** "+str(searchDocs[0].page_content))
    logger.info(" len of searchDocs is "+ len(searchDocs))
    context = ""
    for i in range(min(4,len(searchDocs))):
        context += searchDocs[i].page_content
    return context


"""

prompt = question
messages = [
    {"role": "system", "content": "Answer question by identifying relation and facts in knowledge -  "+context},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=700,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("\n \n the response generated is \n\n", response)
print("\n\n The knowledge used was \n\n", context)

Output on console


***** Printing the first doc split****
 page_content='MediaPipe: A Framework for Building Perception Pipelines
Camillo Lugaresi, Jiuqiang Tang, Hadon Nash, Chris McClanahan, Esha Uboweja, Michael Hays,
Fan Zhang, Chuo-Ling Chang, Ming Guang Yong, Juhyun Lee, Wan-Teh Chang, Wei Hua,
Manfred Georg and Matthias Grundmann
Google Research
mediapipe@google.com
Abstract
Building applications that perceive the world around
them is challenging. A developer needs to (a) select and
develop corresponding machine learning algorithms and
models, (b) build a series of prototypes and demos, (c) bal-
ance resource consumption against the quality of the so-
lutions, and ﬁnally (d) identify and mitigate problematic
cases. The MediaPipe framework addresses all of these
challenges. A developer can use MediaPipe to build pro-
totypes by combining existing perception components, to
advance them to polished cross-platform applications and
measure system performance and resource consumption on
target platforms. We show that these features enable a de-' metadata={'source': 'Mediapipe.pdf', 'page': 0}
Some weights of the model checkpoint at Alibaba-NLP/gte-multilingual-base were not used when initializing NewModel: ['classifier.bias', 'classifier.weight']
- This IS expected if you are initializing NewModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing NewModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).

 Printing result of query result
 [-0.0700540691614151, 0.03607219457626343, -0.031630437821149826]


 Enter query
What is the implementation?

 *****, Q was ,  What is the implementation? ***answer is within doc split **
 from a GraphConfig , each subgraph node is replaced by
the corresponding graph of calculators. As a result, the se-
mantics and performance of the subgraph is identical to the
corresponding graph of calculators.
GraphConfig has several other ﬁelds to conﬁgure
the global graph-level settings, e.g., graph executor con-
ﬁgs, number of threads, and maximum queue size of input
streams. Several graph-level settings are useful for tuning
the performance of the graph on different platforms ( e.g.,
desktop v.s. mobile). For instance, on mobile, attaching a
heavy model-inference calculator to a separate executor can
improve the performance of a real-time application since
this utilizes thread locality.
4. Implementation
This section discusses MediaPipe’s scheduling logic and
powerful synchronization primitives to process time-series
in a customizable fashion.
4.1. Scheduling
4.1.1 Scheduling mechanics
Data processing in a MediaPipe graph occurs inside Calcu-


 len of searchDocs is  4




 the response generated is

 Based on the provided context, the implementation refers to the MediaPipe scheduling logic and powerful synchronization primitives used to process time-series in a customizable manner within a MediaPipe graph. Specifically:

1. Data processing: The Data Processing step occurs inside Calcu-results, merging detections from previous frames and removing duplicates based on location and class proximity.

2. Visualization: The visualization step involves adding overlays to the camera frames with annotations representing the merged detections, and handling synchronization between the annotations and camera frames.

3. Output generation: During a single graph run, the framework constructs calcalculator objects corresponding to a graph node and calls Open(), Process(), and Close() methods on these objects as described in Section 3.4.

4. Graph termination: If the graph fails the validation step, the code handles it accordingly.

5. Parallelization: The parallelization mechanism uses the specification of executors in the pipeline's graph configuration (Section 3.6).

6. Event-driven execution: It supports event-driven execution, allowing for the creation of multiple concurrent tasks that execute at different times.

So in summary, the implementation includes the scheduling logic, visualization, output generation, and event-driven execution. It uses powerful synchronization primitives to handle data processing and visualization, along with event-driven execution to manage the flow of tasks.


 The knowledge used was

 from a GraphConfig , each subgraph node is replaced by
the corresponding graph of calculators. As a result, the se-
mantics and performance of the subgraph is identical to the
corresponding graph of calculators.
GraphConfig has several other ﬁelds to conﬁgure
the global graph-level settings, e.g., graph executor con-
ﬁgs, number of threads, and maximum queue size of input
streams. Several graph-level settings are useful for tuning
the performance of the graph on different platforms ( e.g.,
desktop v.s. mobile). For instance, on mobile, attaching a
heavy model-inference calculator to a separate executor can
improve the performance of a real-time application since
this utilizes thread locality.
4. Implementation
This section discusses MediaPipe’s scheduling logic and
powerful synchronization primitives to process time-series
in a customizable fashion.
4.1. Scheduling
4.1.1 Scheduling mechanics
Data processing in a MediaPipe graph occurs inside Calcu-results and merges them with detections from earlier frames
removing duplicate results based on their location in the
frame and/or class proximity.
Note, that the detection-merging node operates on the
same frame that the new detections were derived from. This
is automatically handled by the default input policy in this
node as it aligns the timestamps of the two sets of detection
results before they are processed together (see Section 4.1.2
for more information). The node also sends merged detec-
tions back to the tracker to initialize new tracking targets if
needed.
For visual display, the detection-annotation node adds
overlays with the annotations representing the merged de-
tections on top of the camera frames, and the synchroniza-
tion between the annotations and camera frames is automat-
ically handled by the default input policy before drawing
takes place in this calculator. The end result is a slightly
delayed viewﬁnder output ( e.g., by a few frames) that issource.
2. The type of an input stream/side packet must be com-
patible with the type of the output stream/side packet
to which it is connected.
3. Each node’s connections are compatible with its con-
tract.
The function returns an error if the graph fails the validation
step.
During a single graph run, the framework constructs cal-
culator objects corresponding to a graph node and calls
Open() ,Process() andClose() methods on these
objects as discussed in Section 3.4. The graph can stop run-
ning when:
•Calculator::Close() has been called on all cal-
culators, or
•All source calculators indicate that they have ﬁnished
sending packets and all graph input streams have been
closed, or
•Any error occurs (the graph returns an error with a
message in this case).
2A use case here is a media decoder reaching the end of ﬁle but still
having additional images in its encoding state.
3on parallel threads with the speciﬁcation of executors in
the pipeline’s graph conﬁguration (refer to Section 3.6 and
4.1.1). In the detection branch, a frame-selection node ﬁrst
selects frames to go through detection based on limiting fre-
quency or scene-change analysis, and passes them to the
detector while dropping the irrelevant frames. The object-
detection node consumes an ML model and the associated
label map as input side packets, performs ML inference
on the incoming selected frames using an inference engine
(e.g., [12] or [2]) and outputs detection results.
In parallel to the detection branch, the tracking branch
updates earlier detections and advances their locations to
the current camera frame.
After detection, the detection-merging node compares
7

"""
