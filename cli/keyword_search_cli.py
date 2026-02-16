#!/usr/bin/env python3

import argparse
from internal.json_search import json_read

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            result = json_read("data/movies.json", args.query)
            
            i = 1
            for film in result:
                print(f"{i}. {film["title"]}")
                i += 1

            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()