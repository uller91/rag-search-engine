import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer
from internal.process_files import get_movies
from internal.chunking import semantic_chunk_command

CACHE_DIR = "cache"
SEARCH_LIMIT = 5
PRINT_LIMIT = 100

class SemanticSearch:
    def __init__(self, model_name = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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

        if os.path.exists(self.embeddings_path):
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

    def search(self, query, limit):
        if len(self.embeddings) == 0 or len(self.documents) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        embedding_query = self.generate_embedding(query)
        
        cos_sim_calc = []
        for i in range(len(self.documents)):
            similarity = cosine_similarity(embedding_query, self.embeddings[i])
            cos_sim_calc.append((similarity, self.documents[i]))

        sorted_cos_sim_calc = sorted(cos_sim_calc, key=lambda x: x[0], reverse=True)

        result = []
        for j in range(limit):
            return_doc = {}
            #print(sorted_cos_sim_calc[j])
            return_doc["score"] = sorted_cos_sim_calc[j][0]
            return_doc["title"] = sorted_cos_sim_calc[j][1]["title"]
            return_doc["description"] = sorted_cos_sim_calc[j][1]["description"]
            result.append(return_doc)

        return result

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

        self.chunk_embeddings_path = os.path.join(CACHE_DIR, "chunk_embeddings.npy")
        self.chunk_metadata_path = os.path.join(CACHE_DIR, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents):
        self.documents = documents

        document_list = []
        for document in documents:
            self.document_map[document["id"]] = document
            document_list.append(f"{document['title']}: {document['description']}")

        chunks_all = []
        chunks_metadata = []

        doc_idx = 0
        for document in self.documents:
            if len(document['description']) == 0:
                continue
            
            chunks = semantic_chunk_command(document['description'], 4, 1)

            chunk_idx = 0
            for chunk in chunks:
                chunks_all.append(chunk)
                metadata = {
                    "movie_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunksear": len(chunks)
                }
                chunks_metadata.append(metadata)
                chunk_idx += 1

            doc_idx += 1

        self.chunk_embeddings = self.model.encode(chunks_all, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata

        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.chunk_embeddings_path, 'wb') as ce:
            np.save(ce, self.chunk_embeddings)
        
        with open(self.chunk_metadata_path, 'w') as cm:
            json.dump({"chunks": chunks_metadata, "total_chunks": len(chunks_all)}, cm, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        document_list = []
        for document in documents:
            self.document_map[document["id"]] = document
            document_list.append(f"{document['title']}: {document['description']}")

        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.chunk_metadata_path):
            with open(self.chunk_embeddings_path, 'rb') as ce:
                self.chunk_embeddings = np.load(ce)

            with open(self.chunk_metadata_path, 'r') as cm:
                data = json.load(cm)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunked(self, query: str, limit: int = 5):      
        if len(self.chunk_embeddings) == 0 or len(self.documents) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_chunk_embeddings` first.")

        embedding_query = self.generate_embedding(query)
        chunk_scores = []
        for i in range(len(self.chunk_embeddings)):
            similarity = cosine_similarity(embedding_query, self.chunk_embeddings[i])
            score = {
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                    "score": similarity
                }
            chunk_scores.append(score)

        scores = {}
        for score in chunk_scores:
            if (score["movie_idx"] not in scores.keys()) or (score["score"] > scores[score["movie_idx"]]):
                scores[score["movie_idx"]] = score["score"]

        ###
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        result = []
        for j in range(limit):
            return_doc = {}
            return_doc["id"] = sorted_scores[j][0]
            return_doc["title"] = self.documents[sorted_scores[j][0]]["title"]
            return_doc["description"] = self.documents[sorted_scores[j][0]]["description"][:PRINT_LIMIT]
            return_doc["score"] = sorted_scores[j][1]
            return_doc["metadata"] = self.chunk_metadata

            result.append(return_doc)

        return result

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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)