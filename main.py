from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI

# Open the file
with open('shakespeare.txt', 'r') as file:
    # Read the content of the file
    body = file.read()

def get_text_chunks_langchain(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(text)
    return docs

documents = get_text_chunks_langchain(body)
# split the documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.create_documents(documents)
# select which embeddings we want to use
embeddings = OpenAIEmbeddings(openai_api_key="<<The openai 3.5 api key>>")
# create the vectorestore to use as the index
db = Chroma.from_documents(texts, embeddings)
# expose this index in a retriever interface
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":5})
# create a chain to answer questions
qa = RetrievalQA.from_chain_type(
  # chain_type has four options, it directly affect the answer you get from the document
  # https://python.langchain.com/en/latest/modules/chains/index_examples/question_answering.html
    llm=OpenAI(openai_api_key="<<The openai 3.5 api key>>"), chain_type="refine", retriever=retriever, return_source_documents=True)
query = "<<<The Question You Will ASK>>"
result = qa({"query": query})

# map reduce
print(result['result'])
