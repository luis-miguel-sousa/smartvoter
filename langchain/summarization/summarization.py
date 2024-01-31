import chromadb
from chromadb.utils import embedding_functions
import json 

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate


def instantiate_collection(collection_name:str = "portuguese_mililm_l6_v2",
                           embedding_function:str = "all-MiniLM-L6-v2" ):
    
    """instantiates chroma client with embedding function + adds a collection"""
    
    chroma_client = chromadb.Client()
    client = chromadb.PersistentClient(path="chroma_db/")
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_function)
    
    collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function=sentence_transformer_ef)
    
    return collection

def get_data(add_to_collection:bool = True):
    
    """get two json files and add to collection"""

    def get_livre_data():

        data = []

        for i in range(1,8):
            f = open(f'livre-2024_{i}.json') 
            loaded = json.load(f)
            data.append(loaded)
        
        data = [x for xs in data for x in xs]
        #data = flatten(data)
        
        #split up docs
        documents = [i['content'] for i in data]
        metadatas = [{'chapter':i['chapter'], 'party':'livre', 'year':2024} for i in data]
        
        return documents, metadatas
    
    def get_il_data():
        documents = []
        metadatas = []
        
        f = open(f'programs/json/il-2022.json') 
        data = json.load(f)

        for j, i in enumerate(data):
            c = 0
            if i['content'] == None:
                c += 1
                pass
            else:
                # only added content here
                content = ' '.join(i['content'])
                chapter = i['chapter']
                party = 'il'
                year = 2022
                
                documents.append(content)
                metadatas.append({'chapter':chapter,\
                    'party':party,\
                        'year':2022})
                
        return documents, metadatas
    
    documents_livre, metadatas_livre = get_livre_data()
    documents_il, metadatas_il = get_il_data()

    full_docs = documents_livre + documents_il
    full_metadata = metadatas_livre + metadatas_il
    full_ids = [str(i) for i in range(len(full_docs))]
    
    print(len(full_docs), len(full_metadata), len(full_ids))
    
    if add_to_collection:
        #add to chroma
        
        collection = instantiate_collection()
        
        collection.add(
            documents=full_docs,
            metadatas=full_metadata,
            ids=full_ids)
        
        print('collection updated')
        
        return None
    
    return full_docs, full_metadata, full_ids

def query_collection(query:str, n_results:int=5):
    
    """queries chroma db and returns n_results of relevant documents"""
    
    collection = instantiate_collection()
    
    results = collection.query(
    query_texts=[query],
        n_results=n_results
    )
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.create_documents(results["documents"][0], metadatas=results['metadatas'][0])
    texts = text_splitter.split_documents(texts)

    return texts

def langchain_client(texts):
    
    """summarizes chroma texts with LLM"""
    
    prompt_template = """You will receive documents in Portuguese. Summarize the documents in Portuguese in bullet points.
              ```{text}```
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["page_content"])
    
    
    local_path = (
        "./llm_model/gpt4all-falcon-q4_0.gguf"  # replace with your desired local file path
    )
    
    # Callbacks support token-wise streaming
    callbacks = [StreamingStdOutCallbackHandler()]

    # Verbose is required to pass to the callback manager
    llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)
    
    stuff_chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
    
    try:
        return stuff_chain.run(texts)
    except Exception as e:
        return print(
            "The code failed since it won't be able to run inference on such a huge context and throws this exception: ",
            e,
        )
        
if __name__ == '__main__':
    
    #only add if it's the first run to add documents to vector db
    get_data(add_to_collection = True)
    
    query = input('Type your query: ')
    
    texts = query_collection(query = query, n_results = 5)
    
    langchain_client(texts)