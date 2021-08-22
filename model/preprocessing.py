import os
from dotenv import load_dotenv
from haystack.preprocessor.utils import convert_files_to_dicts
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor import PreProcessor
from haystack.document_store import FAISSDocumentStore
from model.retriever import retriever
from haystack.file_converter.txt import TextConverter


# Load environment variables

load_dotenv()

'''encode documents'''
def preprocess(dir_file, pattern, delete_all_document):

    load_docs = load_document(dir_file,pattern)

    # Optimize and write down to database (SQLite3)
    # if you want to delete all document_store before : delete_all_document = True

    document_store = store_documents(load_docs,
            delete_all_document=delete_all_document)

    retrieve = retriever(document_store)

    # # Update embedding (only for DPR)

    update_embeddings(document_store=document_store,
                      retriever=retrieve)

    file_name_embeddings = 'embedding'
    document_store.save(file_name_embeddings)

    return "Encode Successful"



def load_document(dir_file, pattern):

    preprocessor = PreProcessor(
        clean_empty_lines=True,
        clean_whitespace=True,
        clean_header_footer=True,
        split_by='word',
        split_length=500,
        split_respect_sentence_boundary=True,
        split_overlap=0,
        )
    
    if pattern == "folder":
        all_docs = convert_files_to_dicts(dir_path=dir_file,clean_func=clean_wiki_text, split_paragraphs=True)
        nested_docs = [preprocessor.process(d) for d in all_docs]
        docs = [d for x in nested_docs for d in x]

        return docs
        
    converter = TextConverter(remove_numeric_tables=True)
    doc_txt = converter.convert(file_path=dir_file, meta=None)

    docs = preprocessor.process(doc_txt)

    return docs



def store_documents(load_docs, delete_all_document):
    """
        To store preprocessed document into database
        To install sqlite3 on Ubuntu, type: 
            $ sudo apt install sqlite3
        Then
            $ sqlite3 data/processed_documents.db
        Enter `.tables` inside SQLite3 prompt then Ctrl + D
    """

    document_store = FAISSDocumentStore(
        sql_url=os.getenv('PROCESSED_DOCUMENTS_DB'),
        faiss_index_factory_str='Flat',
        vector_dim=768,
        return_embedding=True,
        similarity='dot_product',
        index='document',
        duplicate_documents='overwrite',
        )

    if delete_all_document == True:
        document_store.delete_all_documents()

    # Write down processed document into database
    document_store.write_documents(load_docs, index='document')

    return document_store


def update_embeddings(document_store, retriever):
    """
        To compute the Document embeddings
        which will be compared against the Query embedding.
        This step is computationally intensive
        since it will engage the transformer based encoders.
        Having GPU acceleration will significantly speed this up.
    """

    document_store.update_embeddings(retriever=retriever,
            index='document')
