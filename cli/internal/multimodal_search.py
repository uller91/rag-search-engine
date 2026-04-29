from PIL import Image
from sentence_transformers import SentenceTransformer
from internal.process_files import get_movies
from internal.semantic_search import cosine_similarity

PRINT_LIMIT = 100

class MultimodalSearch:
    def __init__(self, documents, model_name="clip-ViT-B-32"):
        #Load a pretrained Sentence Transformer model
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = self.generate_texts()
        self.text_embeddings = self.model.encode(self.texts)
        self.image_embeddings = ""

    def generate_texts(self):
        texts = []
        for document in self.documents:
            text = f"{document['title']}: {document['description']}"
            texts.append(text)

        return texts

    def embed_image(self, path):
        image = Image.open(path)
        image_embeddings = (self.model.encode([image]))[0]
        self.image_embeddings = image_embeddings

    def search_with_image(self, path):
        self.embed_image(path)
        #print(self.image_embeddings)

        cos_sim_calc = {}
        for i in range(len(self.texts)):
            similarity = cosine_similarity(self.image_embeddings, self.text_embeddings[i])
            cos_sim_calc[self.documents[i]["id"]] = {
                "id": self.documents[i]["id"],
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
                "similarity": similarity
            }

        sorted_cos_sim_calc = sorted(cos_sim_calc.items(), key=lambda item: item[1]["similarity"], reverse=True)
        return sorted_cos_sim_calc[:5]

def image_search_command(path):
    movies = get_movies()
    multimodal = MultimodalSearch(movies)
    results = multimodal.search_with_image(path)

    i = 1
    for result in results:
        print(f"{i}. {result[1]["title"]} (similarity: {result[1]["similarity"]:.3f})")
        print(f"    {result[1]["description"][:PRINT_LIMIT]}...")

        i += 1

#not updated to the new constructor
def verify_image_embedding(path):
    movies = get_movies()
    multimodal = MultimodalSearch(movies)
    multimodal.embed_image(path)
    print(f"Embedding shape: {multimodal.image_embeddings.shape[0]} dimensions")