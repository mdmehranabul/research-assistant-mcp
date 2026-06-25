from mcp.server.fastmcp import FastMCP
from pathlib import Path
from typing import List, Set, Dict, Any
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import os
import shutil
import hashlib
import json

mcp = FastMCP("research Assistant")
RESEARCH_DB_PATH = os.getenv("RESEARCH_DB_PATH")
if not RESEARCH_DB_PATH:
    raise ValueError("RESEARCH_DB_PATH environment variable is required. Please set it in your configuration")

CHROMA_DB_ROOT = Path(RESEARCH_DB_PATH)/"research_chroma_dbs"

EMBED_MODEL = "nomic-embed-text"
OLLAMA_BASE_URL = "http://localhost:11434"

embeddings = OllamaEmbeddings(model = EMBED_MODEL, base_url= OLLAMA_BASE_URL)

def get_content_hash(content: str) -> str:
    """ Generate a hash for content to check for duplicates """
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_content_hashes(topic_path: Path) -> Set[str]:
    """Load existing content hashes from metadata file."""
    metadata_file = topic_path / "content_hashes.json"
    if metadata_file.exists():
        try:
            with open(metadata_file,'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def save_content_hashes(topic_path: Path, hashes: Set[str]):
    """Save content hashes to metadata file."""
    metadata_file = topic_path/"content_hashes.json"
    with open(metadata_file,'w') as f:
        json.dump(list(hashes),f)

def get_vectorstore(topic: str) -> Chroma:
    """Get or create a ChromaDB vectorstore for a topic"""
    topic_path = CHROMA_DB_ROOT/topic
    topic_path.mkdir(parents=True, exist_ok=True)

    return Chroma(
        persist_directory=str(topic_path),
        embedding_function=embeddings,
        collection_name=f"research_{topic}"
    )


@mcp.tool()
def save_research_data(content: List[str], topic:str="default") -> str :
    """
    Save research content to vector database for future retrieval
    Args:
        content: List of text content
        topic: Topic name for organizing the data (creates separate DB)
    """
    try:
        topic = topic.replace(' ','_')
        topic_path = CHROMA_DB_ROOT/topic
        topic_path.mkdir(parents=True, exist_ok=True)

        existing_hashes = load_content_hashes(topic_path)

        new_content = []
        new_hashes = set(existing_hashes)

        for text in content:
            content_hash = get_content_hash(text)
            if content_hash not in existing_hashes:
                new_content.append(text)
                new_hashes.add(content_hash)

        if not new_content:
            return f"No new content to save - all {len(content)} documents already exists in topic: {topic}"
        
        vectorstore = get_vectorstore(topic)

        documents = []
        doc_ids = []

        for i, text in enumerate(new_content):
            content_hash = get_content_hash(text)
            doc = Document(
                page_content=text,
                metadata = {
                    "topic": topic,
                    "content_hash":content_hash,
                    "doc_index":len(existing_hashes)+i
                }
            )
            documents.append(doc)
            doc_ids.append(f"{topic}_{content_hash}")
    
        vectorstore.add_documents(documents=documents, ids = doc_ids)

        save_content_hashes(topic_path, new_hashes)

        return f"Successfully saved {len(new_content)} new documents to topic: {topic} (skipped) {len(content) - len(new_content)}"
    
    except Exception as e:
        return f"Error saving research data: {str(e)}"
    
        
@mcp.tool()
def search_research_data(query:str, topic: str = "default", max_results: int = 5) -> str:
    """
    Search through saved research data using semantic similarity.
    Args:
        query: Search query
        topic: Topic database to search in
        max_results: Maximum number of results to return
    """
    try:
        topic_path = CHROMA_DB_ROOT / topic
        if not topic_path.exists():
            return f"No research data found for topic: {topic}"
        
        vectorstore = get_vectorstore(topic)

        try:
            collection = vectorstore._collection
            count = collection.count()
            if count == 0:
                return f"No documents found in topic: {topic}"
        except:
            return f"No research data found for topic: {topic}"
        
        results = vectorstore.similarity_search_with_score(query,k = max_results)
        
        if not results:
            return f"No relevant results found for query: '{query}' in topic: {topic}"
        
        formatted_results = []
        for i, (doc,score) in enumerate(results):
            similarity = 1 - score
            result_text = f"Result {i+1} (Similarity: {similarity: .3f}):\n{doc.page_content}\n"
            formatted_results.append(result_text)
        
        return "\n" + "="*50 + "\n".join(formatted_results) + "="*50
    
    except Exception as e:
        return f"Error searching research data: {str(e)}"

@mcp.tool()
def list_research_topics()->str:
    """
    List all available research topics (vector databases).
    """
    try:
        if not CHROMA_DB_ROOT.exists():
            return "No research topics found"
        
        topics = []
        for path in CHROMA_DB_ROOT.iterdir():
            if path.is_dir():
                try:
                    vectorstore = get_vectorstore(path.name)
                    collection = vectorstore._collection
                    doc_count = collection.count()
                    topics.append(f"Topic: {path.name} ({doc_count} documents)")
                
                except Exception as e:
                    try:
                        hashes = load_content_hashes(path)
                        doc_count = len(hashes)
                        topics.append(f"Topic: {path.name} ({doc_count})")
                    except:
                        topics.append(f"Topic: {path.name}")
        
        if not topics:
            return "No research topics found"
        
        return "\n".join(topics)
    except Exception as e:
        return f"Error listing topics: {str(e)}"


@mcp.tool()
def delete_research_topic(topic: str)->str:
    """
    Delete a research topic and all of its data.
    Args:
        topic: Topic name to delete
    """

    try:
        topic_path = CHROMA_DB_ROOT / topic

        if not topic_path.exists():
            return f"Topic '{topic}' does not exist"
        
        try:
            vectorstore = get_vectorstore(topic)
            vectorstore.delete_collection()
        except :
            pass

        shutil.rmtree(topic_path)
    
        return f"Successfully deleted topic: {topic}"
    
    except Exception as e:
        return f"Error deleting topic: {str(e)}"


@mcp.tool()
def get_topic_info(topic:str)->str:
    """
    Get detailed information about a research topic.
    Args:
        topic: Topic name to get info for
    """
    try:
        topic_path = CHROMA_DB_ROOT / topic

        if not topic_path.exists():
            return f"Topic '{topic}' does not exist"
    
        vectorstore = get_vectorstore(topic)
        collection = vectorstore._collection
        doc_count = collection.count()

        hashes = load_content_hashes(topic_path)
        hash_count = len(hashes)

        info = f"""Topic Information: {topic},
                - ChromaDB Collection: research_{topic}
                - Document Count: {doc_count}
                - Hash records: {hash_count} 
                - Database path: {topic_path}
                - Embedding model: {EMBED_MODEL},
                - Ollama URL: {OLLAMA_BASE_URL}
                """
        return info
    
    except Exception as e:
        return f"Error getting topic info: {str(e)}"

def main():
    """Main entry point for the Research Assistant MCP Server"""
    mcp.run(transport = "stdio")

if __name__ == "__main__":
    main()