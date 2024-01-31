from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import chromadb
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma


def get_texts():
    # load json file
    loader = JSONLoader(
        file_path='programs/json/il-2022.json',
        jq_schema='.[].content',
        text_content=False)

    data = loader.load()
    
    #split json file for llm
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                            chunk_overlap=50)
    texts = splitter.split_documents(data)
    
    return texts

def fill_vector_store():
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'})
    
    texts = get_texts()
    
    db = Chroma.from_documents(texts, embeddings)
    
    return db
        

def load_llm():
    """load the llm"""

    llm = CTransformers(model='llm_model/llama-2-7b-chat.ggmlv3.q2_K.bin', # model available here: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
                    model_type='llama',
                    config={'max_new_tokens': 256, 'temperature': 0})
    return llm

def load_vector_store():
    # load the vector store
    
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     model_kwargs={'device': 'cpu'})
    
    db = fill_vector_store()
    
    #db = FAISS.load_local("faiss", embeddings)
    
    return db

def instantiate_collection(collection_name:str = "portuguese_mililm_l6_v2",
                           embedding_function:str = "all-MiniLM-L6-v2" ):
    
    """instantiates chroma client with embedding function + adds a collection"""
    
    chroma_client = chromadb.Client()
    client = chromadb.PersistentClient(path="chroma_db/")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_function)
    
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=sentence_transformer_ef)
    
    return collection

def create_prompt_template():
    # prepare the template that provides instructions to the chatbot

    template = """Use the provided context to answer the user's question.
    If you don't know the answer, respond with "I do not know".
    Context: {context}
    Question: {question}
    Answer:
    """

    prompt = PromptTemplate(
        template=template,
        input_variables=['context', 'question'])
    return prompt

def create_qa_chain():
    """create the qa chain"""

    # load the llm, vector store, and the prompt
    llm = load_llm()
    db = load_vector_store()
    prompt = create_prompt_template()


    # create the qa_chain
    retriever = db.as_retriever(search_kwargs={'k': 2})
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type='stuff',
                                        retriever=retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={'prompt': prompt})
    
    return qa_chain

def generate_response(query, qa_chain):

    # use the qa_chain to answer the given query
    return qa_chain({'query':query})['result']