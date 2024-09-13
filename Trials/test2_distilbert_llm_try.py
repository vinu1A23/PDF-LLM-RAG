import os
import transformers
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer #Replace "your-model-name" with the actual name of your model

model_name = os.getenv("MODEL_NAME")
print(model_name)
model_config_path = os.getenv("MODEL_CONFIG") #Load the


question_answerer = pipeline("question-answering", model='distilbert/distilbert-base-cased-distilled-squad')

context = r"""

Extractive Question Answering is the task of extracting an answer from a text given a question. An example     of a

question answering dataset is the SQuAD dataset, which is entirely based on that task. If you would like to fine-tune

a model on a SQuAD task, you may leverage the examples/pytorch/question-answering/run_squad.py script.

"""

result = question_answerer(question="What is a good example of a question answering dataset?",     context=context)

print(f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}")
