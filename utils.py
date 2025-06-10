import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    HuggingFaceEmbeddings = None
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

if not EMBEDDINGS_MODEL:
    EMBEDDINGS_MODEL = "text-embedding-ada-002"

# Global variables for Pinecone client and index
pinecone_client = None
pinecone_index = None

parser = StrOutputParser()
template = """
Answer the question based on the context below. If you can't answer the question, reply "As per the context provided, I am unable to answer your question. Please try a different question".

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

# Try to use OpenAI embeddings first, fallback to HuggingFace if quota exceeded
embedding_type = "openai"
try:
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    # Test the embeddings with a simple query
    embeddings.embed_query("test")
except Exception as e:
    if "quota" in str(e).lower() or "insufficient_quota" in str(e):
        print("OpenAI quota exceeded, falling back to HuggingFace embeddings...")
        embedding_type = "huggingface"
        if HuggingFaceEmbeddings:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        else:
            print("HuggingFace embeddings not available, installing...")
            import subprocess
            subprocess.run(["pip", "install", "sentence-transformers"], check=True)
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
model_map = {"gpt-4-turbo-preview": 10, "chatgpt-4o-latest": 10, "claude-3-opus-20240229": 10, "claude-3-sonnet-20240229": 10}

def get_video_id(url):
    if "youtube.com/watch" in url:
        video_id = url.split("youtube.com/watch?v=")[-1].split("&")[0]
    elif "youtube.com/live" in url:
        video_id = url.split("youtube.com/live/")[1].split("?")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[1].split("?")[0]
    else:
        video_id = url
    return video_id

def get_name_space(video_id, embedding_type="openai"):
    base_name = video_id.lower().replace("_", "-")
    if embedding_type == "huggingface":
        return f"{base_name}-hf"
    return base_name

def get_name_spaces():
    global pinecone_index
    if pinecone_index is None:
        print("Pinecone index not initialized. Call load() first to initialize.")
        return ("error", "Pinecone index not initialized.")
    try:
        stats = pinecone_index.describe_index_stats()
        # Ensure to use double quotes for dictionary keys if the main string uses triple double quotes
        return stats.get("namespaces", {}) if stats and isinstance(stats, dict) else {}
    except Exception as e:
        print(f"Error getting namespaces from Pinecone: {e}")
        return ("error", f"Error getting namespaces: {str(e)}")

def get_transcript(video_id):

    transcript = None
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except Exception as e:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            for transcript_obj in transcript_list:
                if transcript_obj.is_translatable:
                    try:
                        transcript = transcript_obj.translate('en').fetch()
                        break
                    except Exception as e:
                        print(e)
        except Exception as e:
            print(e)

    if transcript is None:
        return ("error", "No transcript found")
    
    # Handle both dict and FetchedTranscriptSnippet objects
    try:
        if not transcript or len(transcript) == 0:
            return ("error", "Transcript is empty")
        
        if hasattr(transcript[0], 'text'):
            # FetchedTranscriptSnippet objects
            return ("success", ' '.join(item.text for item in transcript if hasattr(item, 'text')))
        else:
            # Dictionary objects
            return ("success", ' '.join(item.get('text', '') for item in transcript if isinstance(item, dict) and 'text' in item))
    except Exception as e:
        return ("error", f"Error processing transcript: {str(e)}")

def upsert_transcript(data, url, name_space):
    global pinecone_index, embeddings, text_splitter

    if pinecone_index is None:
        return ("error", "Pinecone index not initialized in upsert_transcript. Call load() first.")

    try:
        doc = Document(
                page_content=data,
                metadata={"source": url}
            )
        documents = text_splitter.split_documents([doc])

        vector_store = PineconeVectorStore(
            index=pinecone_index, embedding=embeddings, namespace=name_space
        )
        vector_store.add_documents(documents)

    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower() or "quota" in error_msg.lower(): # Usually from embedding service
            return ("error", "API rate limit or quota exceeded. Please wait or check your API usage limits.")
        elif "insufficient_quota" in error_msg: # Usually from embedding service
            return ("error", "API insufficient quota. Please check your billing and usage limits.")
        else:
            return ("error", f"Error upserting transcript to Pinecone: {error_msg}")
    return ("success", name_space)

def load(url):
    global pinecone_client, pinecone_index, embedding_type # Added pinecone_client, pinecone_index

    # API Key Checks (existing)
    if not os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY") == "your-pinecone-api-key-here":
        return ("error", "Please configure your API keys in the .env file. See README.md for instructions.")
    
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here": # Keep this check too
        return ("error", "Please configure your OpenAI API key in the .env file.")

    # Initialize Pinecone client and index if not already done
    if pinecone_index is None:
        if not PINECONE_INDEX_NAME:
            return ("error", "Pinecone index name (PINECONE_INDEX_NAME) is not configured in environment variables.")
        try:
            print("Initializing Pinecone client and index...") # Added for logging
            pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
            pinecone_index = pinecone_client.Index(PINECONE_INDEX_NAME)
            print(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}") # Added for logging
        except Exception as e:
            return ("error", f"Failed to initialize Pinecone client/index: {str(e)}")
    
    video_id = get_video_id(url)
    name_space = get_name_space(video_id, embedding_type)
    
    current_name_spaces_result = get_name_spaces()
    if isinstance(current_name_spaces_result, tuple) and current_name_spaces_result[0] == "error":
        return current_name_spaces_result
    current_name_spaces = current_name_spaces_result

    if name_space not in current_name_spaces:
        print("creating name space: ", name_space)
        status, data = get_transcript(video_id)
        
        if status != "success":
            return ("error", data)
        
        status, message = upsert_transcript(data, url, name_space)

        if status != "success":
            return ("error", message)

    return ("success", name_space)

def generate(model, name_space, question):
    global pinecone_index, embeddings, prompt, parser, model_map

    if pinecone_index is None:
        yield "Error: Pinecone index not initialized. Please load a video first via the /load endpoint."
        return

    if not os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY") == "your-pinecone-api-key-here":
        yield "Error: Please configure your API keys in the .env file. See README.md for instructions."
        return
    
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        yield "Error: Please configure your OpenAI API key in the .env file."
        return
    
    try:
        if model.find("gpt") >= 0:
            model_obj = ChatOpenAI(model=model, streaming=True)
        else:
            model_obj = ChatAnthropic(model_name=model, streaming=True)

        vector_store_instance = PineconeVectorStore(
                index=pinecone_index, embedding=embeddings, namespace=name_space
            )

        if model not in model_map:
            yield f"Error: Unsupported model '{model}'. Supported models: {', '.join(model_map.keys())}"
            return
            
        chain = (
        {"context": vector_store_instance.as_retriever(k=model_map[model]), "question": RunnablePassthrough()}
            | prompt
            | model_obj
            | parser
        )

        for chunk in chain.stream(question):
            yield chunk
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg or "quota" in error_msg.lower():
            yield "Error: OpenAI API quota exceeded. Please add credits to your OpenAI account or wait for quota reset. You can also try using Claude models instead of GPT models."
        elif "rate limit" in error_msg.lower():
            yield "Error: OpenAI API rate limit exceeded. Please wait a moment and try again."
        elif "embed_query" in error_msg:
            yield "Error: Unable to process question due to OpenAI embedding service quota limits. Please check your OpenAI account billing or try again later."
        else:
            yield f"Error generating response: {error_msg}"


if __name__ == "__main__":
    # import time
    # start_time = time.time()
    # status, name_space = load("https://www.youtube.com/watch?v=kfrbkm_nmak")
    # end_time = time.time()
    # print("load Elapsed time:", end_time - start_time, "seconds")
    # start_time = time.time()
    # for chunk in generate("chatgpt-4o-latest", get_name_space("java-interview-questions"), "what is the life cycle of servlet?"):
    #     print(chunk)
    # end_time = time.time()
    # print("generate Elapsed time:", end_time - start_time, "seconds")
    pass
