"""List Groq models available to the configured API key."""

import os
import sys

from dotenv import load_dotenv
from groq import Groq


def main() -> int:
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api")
    if not api_key:
        print("Missing GROQ_API_KEY (or groq_api) in your environment or .env file.")
        return 1

    try:
        models = sorted(Groq(api_key=api_key).models.list().data, key=lambda model: model.id)
    except Exception as error:
        print(f"Could not retrieve Groq models: {error}", file=sys.stderr)
        return 1

    if not models:
        print("No models are available for this API key.")
        return 0

    print(f"Models available to this API key ({len(models)}):")
    for model in models:
        context_window = getattr(model, "context_window", "unknown")
        print(f"- {model.id} | context window: {context_window}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())