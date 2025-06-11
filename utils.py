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
    try:
        return Pinecone().Index(PINECONE_INDEX_NAME).describe_index_stats()['namespaces']
    except Exception as e:
        print(f"Error connecting to Pinecone: {e}")
        return {}

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
        if hasattr(transcript[0], 'text'):
            # FetchedTranscriptSnippet objects
            return ("success", ' '.join(item.text for item in transcript))
        else:
            # Dictionary objects
            return ("success", ' '.join(item['text'] for item in transcript))
    except Exception as e:
        return ("error", f"Error processing transcript: {str(e)}")

def upsert_transcript(data, url, name_space):
    try:
        doc = Document(
                page_content=data,
                metadata={"source": url}
            )
        documents = text_splitter.split_documents([doc])
        PineconeVectorStore.from_documents(
            documents, embeddings, index_name=PINECONE_INDEX_NAME, namespace=name_space
        )
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower() or "quota" in error_msg.lower():
            return ("error", "OpenAI API rate limit exceeded. Please wait a moment and try again, or check your API usage limits.")
        elif "insufficient_quota" in error_msg:
            return ("error", "OpenAI API quota exceeded. Please check your billing and usage limits.")
        else:
            return ("error", f"Error processing video: {error_msg}")
    return ("success", name_space)

def load(url):
    global embedding_type
    # Check if API keys are configured
    if not os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY") == "your-pinecone-api-key-here":
        return ("error", "Please configure your API keys in the .env file. See README.md for instructions.")
    
    if not os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") == "your-openai-api-key-here":
        return ("error", "Please configure your OpenAI API key in the .env file.")
    
    video_id = get_video_id(url)
    name_space = get_name_space(video_id, embedding_type)
    name_spaces = get_name_spaces()
    
    if name_space not in name_spaces:
        print("creating name space: ", name_space)
        status, data = get_transcript(video_id)
        
        if status != "success":
            return ("error", data)
        
        status, message = upsert_transcript(data, url, name_space)

        if status != "success":
            return ("error", message)

    return ("success", name_space)

def generate(model, name_space, question):
    # Check if API keys are configured
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
        pinecone = PineconeVectorStore(
                embedding=embeddings, index_name=PINECONE_INDEX_NAME, namespace=name_space
            )
        chain = (
        {"context": pinecone.as_retriever(k=model_map[model]), "question": RunnablePassthrough()}
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
            yield f"Error: {error_msg}"


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
