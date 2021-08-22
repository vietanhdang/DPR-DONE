from haystack.pipeline import ExtractiveQAPipeline
from model.reader import read
import os
from haystack.document_store import FAISSDocumentStore
from model.retriever import retriever

def get_pipeline():
    """
        All the steps of this model will be running here
    """
    document_store = load_exist_embeddings(dir_faiss='embedding')
    
    # Get retriever
    retrieve = retriever(document_store)

    # Get reader
    reader = read()

    # Init pipeline
    pipe = ExtractiveQAPipeline(reader,  retrieve)
    
    return pipe

def load_exist_embeddings(dir_faiss):

    document_store = FAISSDocumentStore.load(faiss_file_path=dir_faiss,
            sql_url=os.getenv('PROCESSED_DOCUMENTS_DB'),
            index='document')

    return document_store

def get_answer(question, pipe):


    prediction = pipe.run(query=question,
                          top_k_retriever=10,
                          top_k_reader=3)

    return prediction
