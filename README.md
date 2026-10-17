# CyberDaily

CyberDaily is a cybersecurity news monitoring and analysis project that scrapes multiple security news feeds, stores relevant articles in a vector database, checks for historical similarity, and uses a Groq-powered agent to summarize the latest findings and extract vulnerability data.

## What this project does

- Scrapes cyber news from multiple RSS feeds
- Deduplicates articles against a local RAG store
- Stores article content in a Chroma vector database
- Finds similar historical stories before summarization
- Uses a Groq LLM to generate a detailed cyber news summary
- Extracts vulnerability entries into JSON format
- Saves the resulting vulnerability data to `Encyclopedia.json`

## Project structure

- `main.py` — orchestrates the full workflow
- `Agent.py` — LLM summarization and vulnerability extraction logic
- `Scraper.py` — RSS feed scraping logic
- `RAG.py` — filtering, storing, and similarity search pipeline
- `RAGkit.py` — embedding model and vector retrieval helpers
- `scheduler.py` — simple daily scheduling check
- `Bundle/` — persisted Chroma vector database files
- `Encyclopedia.json` — output file for extracted vulnerabilities
- `requirments.txt` — Python dependencies
- `.env` — environment variables (if present)

## Requirements

- Python 3.10+
- A Groq API key
- Internet access for RSS scraping

## Installation

1. Open a terminal in the project folder.
2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirments.txt
```

## Configuration

Create a `.env` file in the project root with your API configuration:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

The project also checks for `groq_api` as a fallback environment variable in `Agent.py`, so either naming convention can work if you set it in your shell or `.env`.

## Running the pipeline

Run the main workflow:

```bash
python main.py
```

This will:

1. Scrape recent articles
2. Filter out duplicates from the stored vector database
3. Search for similar historical content
4. Summarize the news with the Groq model
5. Extract vulnerability information
6. Append the results to `Encyclopedia.json`

## Output behavior

- The LLM output is printed in the console and stored in the workflow result
- Extracted vulnerabilities are appended to `Encyclopedia.json`
- The vector database persists under `Bundle/` and is reused on later runs

## Notes

- The project expects exactly 25 news documents in the agent flow. If the scraper does not return that number, the agent may raise a `ValueError`.
- The repository includes a daily scheduling check in `scheduler.py`, but it is a lightweight utility rather than a full production scheduler.
- The dependency file is named `requirments.txt` with a typo. It is the file used by the project.

## Typical use cases

- Daily cyber threat monitoring
- Security research aggregation
- Identifying repeat vulnerabilities and similar incidents
- Building a local, searchable cyber-news knowledge base

## License

This project does not currently declare a license. If you plan to distribute or reuse it, add an explicit license file before publishing.
