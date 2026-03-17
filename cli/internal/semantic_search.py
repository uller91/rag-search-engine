import numpy as np
import os
from sentence_transformers import SentenceTransformer
from internal.process_files import get_movies

CACHE_DIR = "cache"

class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

        self.embeddings_path = os.path.join(CACHE_DIR, "movie_embeddings.npy")

    def build_embeddings(self, documents):
        self.documents = documents

        document_list = []
        for document in documents:
            self.document_map[document["id"]] = document
            document_list.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(document_list, show_progress_bar=True)

        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.embeddings_path, 'wb') as e:
            np.save(e, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents

        document_list = []
        for document in documents:
            self.document_map[document["id"]] = document
            document_list.append(f"{document['title']}: {document['description']}")

        if os.path.exists("cache/movie_embeddings.npy"):
            with open(self.embeddings_path, 'rb') as e:
                self.embeddings = np.load(e)

            if len(self.embeddings) == len(self.documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def generate_embedding(self, text):
        if text == "" or len(text) == text.count(" "):
            raise ValueError("The text is empty!")
            
        embedding = self.model.encode([text])
        return embedding[0]

def embed_query_text(query):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def verify_embeddings():
    semantic = SemanticSearch()
    movies = get_movies()
    embeddings = semantic.load_or_create_embeddings(movies)

    print(f"Number of docs:   {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_text(text):
    semantic = SemanticSearch()
    embedding = semantic.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_model():
    semantic = SemanticSearch()

    print(f"Model loaded: {semantic.model}")
    print(f"Max sequence length: {semantic.model.max_seq_length}")