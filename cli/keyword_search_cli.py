#!/usr/bin/env python3

import argparse
from internal.keyword_search import keyword_search 
from internal.inverted_index import InvertedIndex

def main() -> None:
    index = InvertedIndex() #creating an index for the commands

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

    args = parser.parse_args()
    match args.command:
        case "search":
            try:
                index.load() #load an index from cache
            except Exception as e:
                print(e)
                return

            print(f"Searching for: {args.query}")
            result = keyword_search(args.query, index)
            
            i = 1
            for film_id in result:
                print(f"{i}. {index.docmap[film_id]["title"]}. ID: {index.docmap[film_id]["id"]}")
                i += 1

            pass
        case "build":
            index.build()
            index.save()
        case "tf":
            try:
                index.load() #load an index from cache
            except Exception as e:
                print(e)
                return

            print(f"Searching for the term for the term {args.term} in the movie with {args.id} id")
            tf = index.get_tf(args.id, args.term)
            print(f"Term {args.term} was encointered {tf} times")
        case "idf":
            try:
                index.load() #load an index from cache
            except Exception as e:
                print(e)
                return

            idf = index.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            try:
                index.load() #load an index from cache
            except Exception as e:
                print(e)
                return

            tf_idf = index.get_tf(args.id, args.term) * index.get_idf(args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()