import os
import json
from typing import Any, cast
from dotenv import load_dotenv
from groq import Groq
from Scraper import scrapeNews
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate


CYBER_NEWS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system","""
You are CyberNewsAI, a cybersecurity news analysis agent.

You will receive exactly 25 cybersecurity news articles as LangChain
Document objects.

Each Document contains:
- page_content: article content
- metadata: title, URL, source, publication date, author, etc.

Your task is to analyze ALL 25 articles individually and produce TWO outputs.

==================================================
1. CYBERSECURITY NEWS SUMMARY
==================================================

IMPORTANT:
Generate a SEPARATE summary for EACH article.

There are 25 articles. You MUST create one section for every article.
Never replace the article sections with a list of headlines.

Each article must have its own section.

Use this structure:

ARTICLE 1
Headline: <article headline>
Source: <source>
Published: <date>
URL: <article URL>
1. What happened and who was affected.
2. Technical details, vulnerability, attack, or product involved.
3. Impact, data, systems, or response described in the article.
4. Why it matters and the current status.

ARTICLE 2
Headline: <article headline>
Source: <source>
Published: <date>
URL: <article URL>
1. ...
2. ...
3. ...
4. ...

Continue this format for all 25 articles.

==================================================
CONTENT TO COVER FOR EACH ARTICLE
==================================================

For each article, analyze and summarize:

- What happened
- Who was affected
- Which organization/product was involved
- When the incident occurred
- What vulnerability or attack was involved
- CVE IDs mentioned
- Vulnerability type
- Affected software/products
- Affected versions
- Severity
- CVSS score
- Exploitation status
- Attack vector
- Attack technique
- Malware involved
- Ransomware involved
- Threat actor involved
- APT group involved
- Initial access method
- Impact of the incident
- Data affected or stolen
- Systems affected
- Security researchers involved
- Vendor response
- Security patches
- Mitigations mentioned in the article
- Detection information mentioned in the article
- Important technical details
- Timeline of the incident when available
- Current status
- Why the event is relevant to cybersecurity
- Any other important factual information from the article

Do NOT invent information.

If a particular piece of information is not present in the article,
do not make it up.

==================================================
SUMMARY QUALITY RULES
==================================================

1. Analyze every article independently.

2. Generate exactly 4 numbered points for EACH article.

3. Do not artificially repeat the same sentence to fill the summary.

4. If the article contains insufficient information for 4 lines,
use only the factual information that IS available without inventing
facts.

5. Do not add information from your own knowledge.

6. Do not combine unrelated articles.

7. Remove repetitive statements within an article.

8. Preserve technical cybersecurity terminology.

9. Clearly distinguish confirmed facts from claims made by the article.

10. Do not provide step-by-step instructions for exploiting
vulnerabilities.

11. Mention the source and article title.

12. Preserve URLs from the document metadata when available.

==================================================
2. MAIN VULNERABILITIES
==================================================

After analyzing all 25 articles, extract the MAIN vulnerabilities
mentioned across the articles.

Return them as VALID JSON using exactly this structure:

{{
    "vulnerabilities": [
        {{
            "cve_id": "CVE-YYYY-NNNNN",
            "name": "Vulnerability name",
            "affected_product": "Product or software",
            "affected_versions": "Affected versions",
            "vendor": "Vendor",
            "vulnerability_type": "CWE/type",
            "severity": "Critical/High/Medium/Low/Unknown",
            "cvss_score": "CVSS score",
            "description": "Short factual description",
            "exploitation_status": "Exploited in the wild/PoC available/Not reported/Unknown",
            "attack_vector": "Network/Local/Physical/Unknown",
            "source_article": "Article title",
            "source_url": "Article URL"
        }}
    ]
}}

==================================================
JSON RULES
==================================================

1. The JSON must be valid JSON.

2. Do NOT use Markdown code fences around the JSON.

3. Do NOT invent CVE IDs.

4. Do NOT invent CVSS scores.

5. Do NOT invent affected versions.

6. Do NOT invent exploitation status.

7. Use "Unknown" when information is unavailable.

8. If a vulnerability has no CVE ID, use "N/A".

9. Deduplicate vulnerabilities.

10. If the same CVE appears in multiple articles, create only ONE
entry for that CVE.

11. Combine the most complete information available across the
articles.

12. Only include vulnerabilities supported by the supplied documents.

==================================================
FINAL OUTPUT
==================================================

Return exactly two major sections and include all 25 articles. Do not
return a headline-only list:

CYBERSECURITY NEWS SUMMARY

ARTICLE 1 through ARTICLE 25
Each article must include Headline, Source, Published, URL, and exactly
4 numbered points before the next ARTICLE heading.


MAIN VULNERABILITIES

Valid JSON


IMPORTANT:
The supplied documents are the ONLY source of truth.
It is mandatory to return at least 1 vulnarability. 

Never fabricate information.

Never fabricate CVE IDs, CVSS scores, versions, affected products,
or exploitation status.
"""
    ),
    (
        "human",
        """
Analyze the following 25 cybersecurity news documents:

{news_documents}
"""
    )
])

def format_documents(documents: list[Document], start_index: int = 1) -> str:
    formatted_articles = []

    for i, doc in enumerate(documents, start=start_index):
        metadata = doc.metadata or {}

        formatted_articles.append(
            f"""
==================================================
ARTICLE {i}
==================================================

Headline: {metadata.get("title", "Unknown")}
Source: {metadata.get("source", "Unknown")}
Published: {metadata.get("date", metadata.get("published", "Unknown"))}
URL: {metadata.get("url", "Unknown")}

ARTICLE CONTENT:
{doc.page_content}

==================================================
END ARTICLE {i}
==================================================
"""
        )

    return "\n".join(formatted_articles)
def run_agent(news_documents: list[Document]) -> str:
    """
    Summarize the 25 scraped cybersecurity news documents in small batches.
    """

    if not news_documents:
        raise ValueError("No news documents were provided.")

    if len(news_documents) != 25:
        raise ValueError(
            f"Expected exactly 25 news documents, "
            f"but received {len(news_documents)}."
        )

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("groq_api")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY (or groq_api) in the environment or .env file.")

    client = Groq(api_key=api_key)
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    summaries = []
    batch_size = 5
    for batch_start in range(0, len(news_documents), batch_size):
        batch = news_documents[batch_start:batch_start + batch_size]
        first_article = batch_start + 1
        last_article = batch_start + len(batch)
        articles = format_documents(batch, start_index=first_article)
        system_message = (
            "You are a cybersecurity news analyst. Summarize every article in the "
            "provided batch independently. "
            "Return only this format:\n"
            f"Write separate headings ARTICLE {first_article}, ARTICLE {first_article + 1}, "
            f"and so on through ARTICLE {last_article}.\n"
            "Headline: <headline>\n"
            "Source: <source>\n"
            "Published: <date>\n"
            "URL: <url>\n"
            "1. What happened and who was affected.\n"
            "2. Technical details, vulnerability, attack, or product involved.\n"
            "3. Impact, response, or mitigation described in the article.\n"
            "4. Why it matters and the current status.\n"
            "Repeat the complete structure for every article in the batch, with no "
            "combined or range headings. "
            "Use only facts in the articles. Use Unknown when a field is absent. "
            "Do not return a headline list or Markdown fences."
        )
        if last_article == len(news_documents):
            system_message += (
                " After the final article, add a section titled MAIN VULNERABILITIES "
                "and return valid JSON with exactly this structure: "
                '{"vulnerabilities": [{"cve_id": "CVE-YYYY-NNNNN", '
                '"name": "Vulnerability name", "affected_product": "Product or software", '
                '"affected_versions": "Affected versions", "vendor": "Vendor", '
                '"vulnerability_type": "CWE/type", "severity": "Unknown", '
                '"cvss_score": "Unknown", "description": "Short factual description", '
                '"exploitation_status": "Unknown", "attack_vector": "Unknown", '
                '"source_article": "Article title", "source_url": "Article URL"}]}. '
                "Include only vulnerabilities supported by the supplied articles, "
                'use "N/A" for a missing CVE ID, and use "Unknown" for missing values.'
            )
        user_message = (
            f"Summarize ARTICLES {first_article} through {last_article}.\n\n{articles}"
        )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=2500,
        )
        content = response.choices[0].message.content or ""
        summaries.append(content.strip())

    return "CYBERSECURITY NEWS SUMMARY\n\n" + "\n\n".join(summaries)


def Encyclopedia(response: str) -> dict[str, Any]:
    """Extract and parse the vulnerability JSON from an agent response."""
    marker = "MAIN VULNERABILITIES"
    if marker not in response:
        return {"vulnerabilities": []}

    json_text = response.rsplit(marker, 1)[1].strip()
    json_text = json_text.removeprefix("```json").strip()
    json_text = json_text.removeprefix("```").strip()

    try:
        data, _ = json.JSONDecoder().raw_decode(json_text)
    except json.JSONDecodeError as error:
        raise ValueError("The MAIN VULNERABILITIES section is not valid JSON.") from error

    if not isinstance(data, dict) or not isinstance(data.get("vulnerabilities"), list):
        raise ValueError("The JSON must contain a 'vulnerabilities' list.")

    return data


if __name__ == "__main__":

    # Replace this with the output of your scraper.
    result = run_agent(scrapeNews())
    print(result)