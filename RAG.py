from RAGkit import Embeder,VectorStorage,Retriver
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def filter_new_news(documents):
    """Return only articles whose URL or title is not already in the RAG."""
    if not documents:
        return []

    vector_store = VectorStorage(collection_name="Newspaper", persist_dir="Bundle")
    if vector_store.collection.count() == 0:
        return list(documents)

    new_documents = []
    for document in documents:
        metadata = document.metadata or {}
        url = metadata.get("url")
        title = metadata.get("title")
        where = {"url": url} if url else {"title": title} if title else None
        existing = vector_store.collection.get(where=where) if where else {"ids": []}

        if not existing.get("ids"):
            new_documents.append(document)
        else:
            print(f"Skipping article already in RAG: {title or url or 'Unknown'}")

    return new_documents


def spliting_Text(doc):
    text_splittr=RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        separators=['\n']
    )
    split_doc=text_splittr.split_documents(doc)
    print(f"{len(doc)} is split into {len(split_doc)} chunks")
    print(f"First Chunk:-\n{split_doc[0]}")
    return split_doc
def storing(documents):
    if not documents:
        return
    chunks=spliting_Text(documents)
    chunk_text=[doc.page_content for doc in chunks]
    embedManager=Embeder()
    Vector=VectorStorage(collection_name="Newspaper",persist_dir='Bundle')
    embeding=embedManager.gen_Embeding(chunk_text)
    Vector.add_Doc(chunks,embeding)


def find_similar_news(documents, score_threshold: float = 0.75):
    """Return existing RAG documents similar to the newly scraped news."""
    if not documents:
        return []

    vector_store = VectorStorage(collection_name="Newspaper", persist_dir="Bundle")
    if vector_store.collection.count() == 0:
        return []

    embed_manager = Embeder()
    retriever = Retriver(vector_store, embed_manager)
    matches = []
    for document in documents:
        query = f"{document.metadata.get('title', '')}\n{document.page_content}"
        matches.extend(retriever.retrive(query, top_k=1, score_threshold=score_threshold))
    return matches
