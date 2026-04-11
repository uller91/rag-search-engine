import os
from dotenv import load_dotenv
from google import genai


def main():
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")

    client = genai.Client(api_key=api_key)
    model_name = "gemma-3-27b-it"
    request = "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."

    response = client.models.generate_content(
        model=model_name,
        contents=request,
        )

    prompt_token_count = response.usage_metadata.prompt_token_count
    candidates_token_count = response.usage_metadata.candidates_token_count

    print(f"User prompt: {request}")
    print(f"Prompt tokens: {prompt_token_count}")
    print(f"Response tokens: {candidates_token_count}")
    print(response.text)

if __name__ == "__main__":
    main()