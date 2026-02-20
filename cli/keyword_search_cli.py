#!/usr/bin/env python3

import argparse
from internal.keyword_search import keyword_search 
from internal.inverted_index import InvertedIndex

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    index_parser = subparsers.add_parser("build", help="Build index for movies")

    args = parser.parse_args()
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = keyword_search(args.query)
            
            i = 1
            for film in result:
                print(f"{i}. {film["title"]}")
                i += 1

            pass
        case "build":
            index = InvertedIndex()
            index.build()
            index.save()
            docs = index.get_documents("merida")
            print(f"First document for token 'merida' = {docs[0]}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()