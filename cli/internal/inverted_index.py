import os
import pickle
import collections
import math
from internal.process_input import input_tokenize
from internal.process_files import get_movies

CACHE_DIR = "cache"
index_path = "cache/index.pkl"
docmap_path = "cache/docmap.pkl"
term_frequencies_path = "cache/term_frequencies.pkl"

BM25_K1 = 1.5
BM25_B = 0.75

class InvertedIndex:
    def __init__(self):
        self.index = {} # [token] = (id...)
        self.docmap = {} # [id] = {film}
        self.term_frequencies = {} # [id] = {Counter(word:times....)}
        self.doc_lengths = {} # [id] = {len(text)}
        
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_DIR, "doc_lengths.pkl")

    def __add_document(self, doc_id, text) -> None:
        tokens = input_tokenize(text)
        tf = collections.Counter()
        for token in tokens:
            tf[token] += 1
            if token in self.index.keys():
                current = self.index[token]
                if doc_id in current:
                    continue
                else:
                    current.add(doc_id)
                    self.index[token] = current
            else:
                self.index[token] = {doc_id}
        self.term_frequencies[doc_id] = tf
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        number = len(self.doc_lengths.values())
        print(number)
        if number == 0:
            return 0.0

        total = 0
        for ln in self.doc_lengths.values():
            total += ln
        print(total)

        return (total / number)

    def get_documents(self, term: str) -> list[str]:
        if term.lower() not in self.index.keys():
            return []
        docs = list(self.index[term.lower()])
        return sorted(docs)

    def get_tf(self, doc_id, term: str) -> int:
        token = input_tokenize(term)
        if len(token) > 1:
            raise Exception("The search term should only have one word!")

        tf = self.term_frequencies[doc_id]
        return tf[token[0]] #Collection() returns 0 if term doesn't exist

    def get_idf(self, term: str) -> float:
        token = input_tokenize(term)
        if len(token) > 1:
            raise Exception("The search term should only have one word!")

        total_num_doc = len(self.docmap)
        term_match_num = 0
        for id in self.docmap:
            tf = self.get_tf(id, token[0])
            if tf != 0:
                term_match_num += 1
        
        return math.log((total_num_doc + 1) / (term_match_num + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = input_tokenize(term)
        if len(token) > 1:
            raise Exception("The search term should only have one word!")

        total_num_doc = len(self.docmap)
        term_match_num = 0
        for id in self.docmap:
            tf = self.get_tf(id, token[0])
            if tf != 0:
                term_match_num += 1
        
        return math.log((total_num_doc - term_match_num + 0.5) / (term_match_num + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        tf_component = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        
        return tf_component

    def build(self) -> None:
        movies = get_movies()

        for film in movies:
            self.docmap[film["id"]] = film
            film_data = f"{film["title"]} {film["description"]}"
            self.__add_document(film["id"], film_data)

        print("index build is finished!")

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, 'wb') as i:
            pickle.dump(self.index, i)

        with open(self.docmap_path, 'wb') as d:
            pickle.dump(self.docmap, d)

        with open(self.term_frequencies_path, 'wb') as tf:
            pickle.dump(self.term_frequencies, tf)

        with open(self.doc_lengths_path, 'wb') as dl:
            pickle.dump(self.doc_lengths, dl)
        
        print("index save is finished!")

    def load(self) -> None:
        try:
            with open(self.index_path, 'rb') as i:
                self.index = pickle.load(i)
        except FileNotFoundError:
            raise Exception("error when trying to read index!")
            return

        try:
            with open(self.docmap_path, 'rb') as d:
                self.docmap = pickle.load(d)
        except FileNotFoundError:
            raise Exception("error when trying to read docmap!")
            return

        try:
            with open(self.term_frequencies_path, 'rb') as tf:
                self.term_frequencies = pickle.load(tf)
        except FileNotFoundError:
            raise Exception("error when trying to read term frequencies!")
            return

        try:
            with open(self.doc_lengths_path, 'rb') as dl:
                self.doc_lengths = pickle.load(dl)
        except FileNotFoundError:
            raise Exception("error when trying to read doc lengths!")
            return
        
        #print("index load is finished!")