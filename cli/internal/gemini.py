import os
import time
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

DESCRIPTION_LIMIT = 100

def load_api():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    return api_key

def summarize(query, results):
    api_key = load_api()

    client = genai.Client(api_key=api_key)
    model_name = "gemma-3-27b-it"

    doc_list = []      

    for result in results:
        #doc_list.append(f"{result[1]["document"]["title"]}")
        doc_list.append(f"{result[1]["document"]["title"]} - {result[1]["document"]["description"][:DESCRIPTION_LIMIT]}")

    doc_list_str = chr(10).join(doc_list)

    request = f"""
    Provide information useful to the query below by synthesizing data from multiple search results in detail.

    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Search results:
    {doc_list_str}

    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""

    response = client.models.generate_content(
        model=model_name,
        contents=request,
        )

    return (response.text).strip()

def rag(query, results):
    api_key = load_api()

    client = genai.Client(api_key=api_key)
    model_name = "gemma-3-27b-it"

    doc_list = []      

    for result in results:
        #doc_list.append(f"{result[1]["document"]["title"]}")
        doc_list.append(f"{result[1]["document"]["title"]} - {result[1]["document"]["description"][:DESCRIPTION_LIMIT]}")

    doc_list_str = chr(10).join(doc_list)

    request = f"""
    You are a RAG agent for Hoopla, a movie streaming service.
    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
    Provide a comprehensive answer that addresses the user's query.

    Query: {query}

    Documents:
    {doc_list_str}

    Answer:"""

    response = client.models.generate_content(
        model=model_name,
        contents=request,
        )

    return (response.text).strip()

def evaluate_results(query, results):    
    api_key = load_api()

    client = genai.Client(api_key=api_key)
    model_name = "gemma-3-27b-it"

    temp_results = []
    doc_list = []      

    for result in results:
        #doc_list.append(f"{result[1]["document"]["title"]}")
        doc_list.append(f"{result[1]["document"]["title"]} - {result[1]["document"]["description"][:100]}")

    doc_list_str = chr(10).join(doc_list)

    request = f"""Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {doc_list_str}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers other than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(
        model=model_name,
        contents=request,
        )

    evaluations = json.loads(response.text)

    i = 0
    for result in results:
        result = list(result)
        result.append(evaluations[i])
        temp_results.append(result)
        i += 1

    return temp_results

def improve_result(command, query, results):
    api_key = load_api()

    client = genai.Client(api_key=api_key)
    model_name = "gemma-3-27b-it"
    
    match command:
        case "batch":
            temp_results = []
            doc_list_str = ""            

            for result in results:
                movies_str = f"{result[1]["document"]["title"]} - Movie ID {result[1]["document"]["id"]} - {result[1]["document"]["description"]}   "
                doc_list_str += movies_str

            request = f"""
            Rank the movies listed below by relevance to the following search query.

            Query: "{query}"

            Movies:
            {doc_list_str}

            Return ONLY the movie IDs in order of relevance (best match first). Include ALL IDs given to you. Return a valid JSON list, nothing else.

            For example:
            [75, 12, 34, 2, 1]

            Ranking:
            """

            response = client.models.generate_content(
                model=model_name,
                contents=request,
                )

            rankings = json.loads(response.text)

            for result in results:
                result = list(result)
                if result[1]["document"]["id"] not in rankings:
                    continue
                rank = 1 + rankings.index(result[1]["document"]["id"]) 
                result.append(rank)
                temp_results.append(result)
            
            results_sorted = sorted(temp_results, key = lambda x: (x[2], x[1]["rrf_score"]), reverse=False)
            return results_sorted

        case "individual":
            temp_results = []

            for result in results:
                request = f"""
                Rate how well this movie matches the search query.

                Query: "{query}"
                Movie: {result[1]["document"]["title"]} - {result[1]["document"]["description"]}

                Consider:
                - Direct relevance to query
                - User intent (what they're looking for)
                - Content appropriateness

                Rate 0-10 (10 = perfect match).
                Output ONLY the number in your response, no other text or explanation.

                Score:
                """

                response = client.models.generate_content(
                    model=model_name,
                    contents=request,
                    )

                #print(f"{result[0]} : {response.text}")
                result = list(result)
                result.append((response.text).strip())
                temp_results.append(result)

                #time.sleep(3)

            results_sorted = sorted(temp_results, key = lambda x: (x[2], x[1]["rrf_score"]), reverse=True)
            return results_sorted

def improve_query(command, query):
    api_key = load_api()

    client = genai.Client(api_key=api_key)
    model_name = "gemma-3-27b-it"

    match command:
        case "spell":
            request = f"""
            Fix any spelling errors in the user-provided movie search query below.
            Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
            Preserve punctuation and capitalization unless a change is required for a typo fix.
            If there are no spelling errors, or if you're unsure, output the original query unchanged.
            Output only the final query text, nothing else.
            User query: "{query}"
            """
        case "rewrite":
            request = f"""
            Rewrite the user-provided movie search query below to be more specific and searchable.

            Consider:
            - Common movie knowledge (famous actors, popular films)
            - Genre conventions (horror = scary, animation = cartoon)
            - Keep the rewritten query concise (under 10 words)
            - It should be a Google-style search query, specific enough to yield relevant results
            - Don't use boolean logic

            Examples:
            - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
            - "movie about bear in london with marmalade" -> "Paddington London marmalade"
            - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

            If you cannot improve the query, output the original unchanged.
            Output only the rewritten query text, nothing else.

            User query: "{query}"
            """
        case "expand":
            request = f"""Expand the user-provided movie search query below with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            Output only the additional terms; they will be appended to the original query.
            Can add humans qualities, famous people or fields of science as synonym

            Examples:
            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted"
            - "math" -> "intellect physics"

            User query: "{query}"
            """

    response = client.models.generate_content(
        model=model_name,
        contents=request,
        )

    #prompt_token_count = response.usage_metadata.prompt_token_count
    #candidates_token_count = response.usage_metadata.candidates_token_count

    #print(f"User prompt: {request}")
    #print(f"Prompt tokens: {prompt_token_count}")
    #print(f"Response tokens: {candidates_token_count}")
    #print(response.text)

    if command == "expand":
        return f"{query} {(response.text).strip()}"

    return (response.text).strip()