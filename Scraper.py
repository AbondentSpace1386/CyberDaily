import feedparser
from langchain_core.documents import Document
def conv_to_Doc(text):
    documents = []
    for item in text:
        documents.append(
            Document(
                    page_content=f"""Title: {item['title']},Date: {item['date']},Summary:{item['summary']}
                            """,
                    metadata={
                        "title": item['title'],
                        "url": item['url'],
                        "date": item['date'],
                        "source": item.get('source', 'Unknown')
                    }
                )
            )
    return documents
def scrapeNews():
    News=[]
    feeds = {
        "IT Security Guru": "https://www.itsecurityguru.org/feed/",
        "The Hacker News": "https://feeds.feedburner.com/TheHackersNews",
        "BleepingComputer": "https://www.bleepingcomputer.com/feed/",
        "SecurityWeek": "https://feeds.feedburner.com/securityweek",
        "Graham Cluley": "https://grahamcluley.com/feed/"
    }

    for name, url in feeds.items():
        feed = feedparser.parse(url)
        for article in feed.entries[:5]:
            Hit = {
                "title": article.get("title"),
                "url": article.get("link"),
                "date": article.get("published"),
                "summary": article.get("summary"),
                "source": name,
            }
            News.append(Hit)
    document=conv_to_Doc(News)
    return document

if __name__ == "__main__":
    a=scrapeNews()
    print(a)