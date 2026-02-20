import os
import pickle
from internal.process_input import input_tokenize
from internal.process_files import get_movies

cache_path = "cache"
index_path = "cache/index.pkl"
docmap_path = "cache/docmap.pkl"

class InvertedIndex:
    def __init__(self):
        self.index = {} # [token] = (id...)
        self.docmap = {} # [id] = {film}

    def __add_document(self, doc_id, text):
        tokens = input_tokenize(text)
        for token in tokens:
            if token in self.index.keys():
                current = self.index[token]
                if doc_id in current:
                    continue
                else:
                    current.add(doc_id)
                    self.index[token] = current
            else:
                self.index[token] = {doc_id}

    def get_documents(self, term) -> list[str]:
        docs = list(self.index[term.lower()])
        return sorted(docs)

    def build(self):
        movies = get_movies()

        for film in movies:
            self.docmap[film["id"]] = film
            film_data = f"{film["title"]} {film["description"]}"
            self.__add_document(film["id"], film_data)

        print("index build is finished!")

    def save(self):
        os.makedirs(cache_path, exist_ok=True)

        with open(index_path, 'wb') as i:
            pickle.dump(self.index, i)

        with open(docmap_path, 'wb') as d:
            pickle.dump(self.docmap, d)
        
        print("index save is finished!")
