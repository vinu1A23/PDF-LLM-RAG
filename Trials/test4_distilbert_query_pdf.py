import os
import transformers
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer #Replace "your-model-name" with the actual name of your model
from langchain_community.document_loaders import PyPDFLoader

model_name = os.getenv("MODEL_NAME")
print(model_name)
model_config_path = os.getenv("MODEL_CONFIG") #Load the


question_answerer = pipeline("question-answering", model='distilbert/distilbert-base-cased-distilled-squad')

file_path = "Mediapipe.pdf" # put the pdf in the folder of the python file itself, we had mediapipe
print("file path given as", file_path)
loader = PyPDFLoader(file_path)

docs = loader.load()

print("len of docs is" ,len(docs))
all_data=""
for i in range(len(docs)):
    all_data+=docs[i].page_content[0:]
print("Printing a bit of all_data ",all_data[:100])

context = all_data
result = question_answerer(question="What is the architecture of Mediapipe?",     context=context)

print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
