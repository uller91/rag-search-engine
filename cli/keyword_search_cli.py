#!/usr/bin/env python3

import argparse
from internal.keyword_search import keyword_search, SEARCH_LIMIT
from internal.inverted_index import InvertedIndex, BM25_K1, BM25_B

def bm25_idf_command(term: str) -> float:
    index = InvertedIndex()
    try:
        index.load() #load an index from cache
    except Exception as e:
        print(e)
        return

    return float(index.get_bm25_idf(term))

def bm25_tf_command(doc_id, term, k1 = BM25_K1, b = BM25_B) -> float:
    index = InvertedIndex()
    try:
        index.load() #load an index from cache
    except Exception as e:
        print(e)
        return

    return float(index.get_bm25_tf(doc_id, term, k1, b))

def bm25search_command(query, limit = SEARCH_LIMIT) -> None:
    index = InvertedIndex()
    try:
        index.load() #load an index from cache
    except Exception as e:
        print(e)
        return

    search_result = index.bm25_search(query, limit)

    i = 1
    for result in search_result:
        print(f"{i}. ({index.docmap[result[0]]["id"]}) {index.docmap[result[0]]["title"]} - Score: {result[1]:.2f}")
        i += 1

    return
        

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build index for movies")

    tf_parser = subparsers.add_parser("tf", help="Returns the term frequency in the film description/title. Usage: tf film_id term")
    tf_parser.add_argument("id", type=int, help="Movie ID")
    tf_parser.add_argument("term", type=str, help="Search term")

    idf_parser = subparsers.add_parser("idf", help="Prints Inverse Document Frequency for the given term. Usage: idf term")
    idf_parser.add_argument("term", type=str, help="Search term")

    tfidf_parser = subparsers.add_parser("tfidf", help="Returns IF-IDF score. Usage: tf film_id term")
    tfidf_parser.add_argument("id", type=int, help="Movie ID")
    tfidf_parser.add_argument("term", type=str, help="Search term")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser(
    "bm25tf", help="Get BM25 TF score for a given document ID and term"
    )
    bm25_tf_parser.add_argument("id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("-l", "--limit", type=int, nargs='?', default=SEARCH_LIMIT, help="Optional BM25 search limit")

    args = parser.parse_args()
    match args.command:
        case "search":
            index = InvertedIndex()
            try:
                index.load() #load an index from cache
            except Exception as e:
                print(e)
                return

            print(f"Searching for: {args.query}")
            search_result = keyword_search(args.query, index)
            
            i = 1
            for film_id in search_result:
                print(f"{i}. {index.docmap[film_id]["title"]}. ID: {index.docmap[film_id]["id"]}")
                i += 1

            return
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
        case "tf":
            index = InvertedIndex()
            try:
                index.load() #load an index from cache
            except Exception as e:
                print(e)
                return

            print(f"Searching for the term for the term {args.term} in the movie with {args.id} id")
            tf = index.get_tf(args.id, args.term)
            print(f"Term {args.term} was encointered {tf} times")
        case "idf":
            index = InvertedIndex()
            try:
                index.load() #load an index from cache
            except Exception as e:
                print(e)
                return

            idf = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            index = InvertedIndex()
            try:
                index.load() #load an index from cache
            except Exception as e:
                print(e)
                return

            tf_idf = index.get_tf(args.id, args.term) * index.get_idf(args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tf_idf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case "bm25tf":
            bm25tf = bm25_tf_command(args.id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.id}': {bm25tf:.2f}")
        case "bm25search":
            print(f"Searching for: {args.query}")
            bm25search_command(args.query, args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()