from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        #Load a pretrained Sentence Transformer model
        self.model = SentenceTransformer(model_name)

    def embed_image(self, path):
        image = Image.open(path)
        self.image_embeddings = (self.model.encode([image]))[0]
        return self.image_embeddings

def verify_image_embedding(path):
    multimodal = MultimodalSearch()
    embedding = multimodal.embed_image(path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")