
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer,  AutoModelForCausalLM
from transformers import AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
import os


cache_directory = os.getenv("cache_directory")

device = 'cpu'
file_path = "Mediapipe.pdf" # put the pdf in the folder of the python file itself, we had mediapipe
loader = PyPDFLoader(file_path)

docs = loader.load()


# Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
# It splits text into chunks of 1000 characters each with a 150-character overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# 'data' holds the text you want to split, split the text into documents using the text splitter.
document_splitted = text_splitter.split_documents(docs)

print("\n***** Printing the first doc split**** \n" ,document_splitted[0])

# Define the path to the pre-trained model you want to use
modelPath = "Alibaba-NLP/gte-multilingual-base"

# Create a dictionary with model configuration options, specifying to use the CPU for computations
model_kwargs = {'device':device, 'trust_remote_code':'True'}

# Create a dictionary with encoding options, specifically setting 'normalize_embeddings' to False
encode_kwargs = {'normalize_embeddings': False}

# Initialize an instance of HuggingFaceEmbeddings with the specified parameters
embeddings =    HuggingFaceEmbeddings(
    model_name= modelPath,     # Provide the pre-trained model's path
    model_kwargs=model_kwargs, # Pass the model configuration options
    encode_kwargs=encode_kwargs, # Pass the encoding options
    cache_folder= cache_directory
)

text = "This is a test document."
query_result = embeddings.embed_query(text)
print("\n Printing result of query result \n" , query_result[:3])

db = FAISS.from_documents(document_splitted, embeddings)


question = ""

if question is None or question == "":
    question = input("\n\n Enter query \n")
else:
    question = "What is architecture used?"

searchDocs = db.similarity_search(question)
print("\n *****, Q was , ", question, "***answer is within doc split **\n",searchDocs[0].page_content)
print("\n\n len of searchDocs is ", len(searchDocs),"\n\n")
context = ""
for i in range(min(4,len(searchDocs))):
    context += searchDocs[i].page_content

# Specify the model name you want to use
model_name = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    torch_dtype="auto",
    device_map="cpu", cache_dir= cache_directory
)
# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512, cache_dir=cache_directory)

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


"""
# Define a question-answering pipeline using the model and tokenizer
question_answerer = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=tokenizer,
    return_tensors='pt',
    cache_dir=cache_directory,
    device = 'cpu'
)

# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
# with additional model-specific arguments (temperature and max_length)
llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={'device':'cpu',"temperature": 0.7, "max_length": 512},
    cache_dir=cache_directory
)

# Create a retriever object from the 'db' with a search configuration where it retrieves up to 4 relevant splits/documents.
retriever = db.as_retriever(search_kwargs={"k": 4})
print(retriever)

# Create a question-answering instance (qa) using the RetrievalQA class.
# It's configured with a language model (llm), a chain type "refine," the retriever we created, and an option to not return source documents.
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="MapReduce", retriever=retriever, return_source_documents=False,
    cache_dir=cache_directory, device = 'cpu')

prompt = "What is architecture used?"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
result = qa.run({"query":prompt})
print(result["result"])

"""
