from langchain_community.document_loaders import PyPDFLoader

file_path = "Mediapipe.pdf" # put the pdf in the folder of the python file itself, we had mediapipe
loader = PyPDFLoader(file_path)

docs = loader.load()
all_data=""
print(len(docs))
print(docs[0].page_content[0:])
print(docs[0].metadata)
for i in range(len(docs)):
    all_data+=docs[i].page_content[0:]
print(all_data)
