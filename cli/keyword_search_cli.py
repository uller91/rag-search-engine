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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()