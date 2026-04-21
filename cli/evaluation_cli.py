import argparse
from internal.process_files import get_golden_dataset, get_movies
from internal.hybrid_search import HybridSearch, RRF_K, SEARCH_LIMIT

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    golden_dataset = get_golden_dataset()
    movies = get_movies()
    hybrid = HybridSearch(movies)

    print(f"k={args.limit}\n")

    for test_case in golden_dataset:
        search_result = hybrid.rrf_search(test_case["query"], RRF_K, args.limit)

        all_titles = []
        relevant_titles = []
        relevant_retrieved = 0
        total_retrieved = len(search_result)
        total_relevant = len(test_case["relevant_docs"])

        for result in search_result:
            all_titles.append(result[1]["document"]["title"])
            if result[1]["document"]["title"] in test_case["relevant_docs"]:
                relevant_retrieved += 1
                relevant_titles.append(result[1]["document"]["title"])

        retrieved_str = ", ".join(all_titles)
        relevant_str = ", ".join(relevant_titles)
        precision = relevant_retrieved / total_retrieved
        recall = relevant_retrieved / total_relevant
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        print(f"- Query: {test_case["query"]}")
        print(f"    - Precision@{args.limit}: {precision:.4f}")
        print(f"    - Recall@{args.limit}: {recall:.4f}")
        print(f"    - F1 Score: {f1_score:.4f}")
        print(f"    - Retrieved: {retrieved_str}")
        print(f"    - Relevant: {relevant_str}")


if __name__ == "__main__":
    main()