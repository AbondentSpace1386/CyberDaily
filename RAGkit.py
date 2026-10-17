import numpy as np
import chromadb # storing
from chromadb.config import Settings
from  sentence_transformers import SentenceTransformer #embeting model
import uuid # fro a unique id of each entry
from typing import List,Dict,Any,Tuple
from sklearn.metrics.pairwise import cosine_similarity # for retrival
import os
os.environ["HF_TOKEN"] = "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"

class Embeder():
    def __init__(self,model_name: str="all-MiniLM-L6-v2"):
        """
                Initialize the embedding manager

                Args:
                    model name from Hugging face for sentence embeding
        """
        self.model_name=model_name
        self.model= None
        self._load_model()
        pass
    def _load_model(self):
        """ Loades the Embeding model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model=SentenceTransformer(self.model_name)
            print(f"Model loaded Successfully...\nEmbeddimng Dimentions:-{self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def gen_Embeding(self,texts: List[str])-> np.ndarray:
        if not self.model:
            raise ValueError("Model Not Loaded")
        print(f"Generating embedding for {len(texts)} texts...")
        embeddings=self.model.encode(texts,show_progress_bar=True)
        print(f"Generate embedding with shape: {embeddings.shape}")
        return embeddings


class VectorStorage():
    def __init__(self,collection_name :str ="pdf_docs", persist_dir :str="data/vetcor_store"):
        """Initialize the Vector store
        Arg:
            COllection name : Chrome DB name
            presist_dir: directory to store vertors 
        """
        self.collection_name=collection_name
        self.persist_dir=persist_dir
        self.client=None
        self.collection=None
        self.initialize_store()
    

    def initialize_store(self):
        """Start the persistant chromeDB and collection"""
        try:
            os.makedirs(self.persist_dir,exist_ok=True)
            self.client=chromadb.PersistentClient(path=self.persist_dir)
            # Collection initialization
            self.collection=self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={'description':"MEDICAL Encyclopedias for RAG"}
            )
            print(f"Vector store initialized.Collection:{self.collection_name}")
            print(f"Exixting documents in collection:{self.collection.count()}")
        except Exception  as e:
            print("Error initialixint Vector Store")
            pass

    def add_Doc(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of docs should match number of embeddings")
        
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        # 1. Prepare all the data first
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())

        batch_size = 5000  # Staying under the 5461 limit
        total_len = len(ids)
        
        try:
            for i in range(0, total_len, batch_size):
                end_idx = min(i + batch_size, total_len)
                
                self.collection.add(
                    ids=ids[i:end_idx],
                    embeddings=embeddings_list[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    documents=documents_text[i:end_idx]
                )
                print(f"Successfully added batch {i} to {end_idx}")

            print(f"Total documents successfully added: {total_len}")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error in adding document to vector store: {e}")
            raise

class Retriver():
    """
        Handles query-based retrival from vector store
    """
    def __init__(self,vector_store: VectorStorage, embedding_manager: Embeder):
        """
            Initialize th retrieve
        """
        self.vector_store=vector_store
        self.embedding_manager=embedding_manager
        pass

    def retrive(self, query: str, top_k:int=5, score_threshold: float=0.0)-> list[dict[str,any]]:
        '''
            Retrive relevent documents for a query

            Arg:
                query: serach query
                top_k: Number of results
                score_threshold: Minimun similarity
            Return:
                List of Directories containing retrieved documents and metadata
        '''
        print(f"Retriveing documents for query:-'{query}'")
        print(f"Top K: {top_k}, Socre threshold: {score_threshold}")

        query_embedding= self.embedding_manager.gen_Embeding([query])[0]

        try:
            result=self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            

            retive_docs=[]
            if result['documents'] and result['documents'][0]:
                documents=result['documents'][0]
                metadatas=result['metadatas'][0]
                distances=result['distances'][0]
                ids=result['ids'][0]
                
                for i, (doc_id,document,metadata,distance) in enumerate(zip(ids,documents,metadatas,distances)):

                    #convert distance to similarity score
                    similarity_score=1-distance

                    if similarity_score>= score_threshold:
                        retive_docs.append({
                            'id':doc_id,
                            'content':document,
                            'metadata':metadata,
                            'similarity_score': similarity_score,
                            'distance':distance,
                            'rank': i+1
                            })
                print(f"Retrived {len(retive_docs)} document (after filtering)")
            else:
                print("No Document found")
            return retive_docs
        except Exception as e:
            print(f"Error during retrieva: {e}")
            return[]

