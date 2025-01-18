# AI-powered-YouTube-Summonizer

# Import necessary libraries for the YouTube bot
import gradio as gr
import re  #For extracting video id 
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes  # For specifying model types
from ibm_watsonx_ai import APIClient, Credentials  # For API client and credentials management
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # For managing model parameters
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods  # For defining decoding methods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings  # For interacting with IBM's LLM and embeddings
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs  # For retrieving model specifications
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes  # For specifying types of embeddings
from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates


def get_transcript(url):
    # Extract the video ID from the URL
    defideo_id = get_video_id(url)
    
    # List available transcripts for the video
    srt = YouTubeTranscriptApi.list_transcripts(video_id)
    
    transcript = ""
    
    # Loop through available transcripts and choose the most suitable one
    for i in srt:
        # If no transcript is chosen yet, and it's auto-generated, fetch it
        if not transcript and i.is_generated:
            transcript = i.fetch()
        # If a manually uploaded transcript exists, prioritize it
        elif not i.is_generated:
            transcript = i.fetch()
    
    # Return the fetched transcript (in list format, with time stamps and text)
    return transcript


def process(transcript):
    # Initialize an empty string to hold the formatted transcript
    txt = ""
    
    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            txt += f"Text: {i['text']} Start: {i['start']}\n"
        except KeyError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
            
    # Return the processed transcript as a single string
    return txt


def get_video_id(url):    
    # Regex pattern to match YouTube video URLs
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    # Initialize the RecursiveCharacterTextSplitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    # Split the transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    return chunks

def setup_credentials():
    # Define the model ID for the WatsonX model being used
    model_id = ModelTypes.GRANITE_13B_CHAT_V2
    
    # Set up the credentials by specifying the URL for IBM Watson services
    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
    
    # Create an API client using the credentials
    client = APIClient(credentials)
    
    # Define the project ID associated with the WatsonX platform
    project_id = "skills-network"
    
    # Return the model ID, credentials, client, and project ID for later use
    return model_id, credentials, client, project_id

def define_parameters():
    # Return a dictionary containing the parameters for the WatsonX model
    return {
        # Set the decoding method to GREEDY for generating text
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        
        # Specify the minimum number of new tokens to generate
        GenParams.MIN_NEW_TOKENS: 1,
        
        # Specify the maximum number of new tokens to generate
        GenParams.MAX_NEW_TOKENS: 500,
        
        # Define stop sequences to indicate when generation should halt
        GenParams.STOP_SEQUENCES: ["\n\n"],
    }

def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
    # Create and return an instance of the WatsonxLLM with the specified configuration
    return WatsonxLLM(
        model_id=model_id.value,          # Set the model ID for the LLM
        url=credentials.get("url"),      # Retrieve the service URL from credentials
        project_id=project_id,            # Set the project ID for accessing resources
        params=parameters                  # Pass the parameters for model behavior
    )

def setup_embedding_model(credentials, project_id):
    # Create and return an instance of WatsonxEmbeddings with the specified configuration
    return WatsonxEmbeddings(
        model_id=EmbeddingTypes.IBM_SLATE_30M_ENG.value,  # Set the model ID for the SLATE-30M embedding model
        url=credentials["url"],                            # Retrieve the service URL from the provided credentials
        project_id=project_id                               # Set the project ID for accessing resources in the Watson environment
    )

def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.
    
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: FAISS index
    """
    # Use the FAISS library to create an index from the provided text chunks
    return FAISS.from_texts(chunks, embedding_model)

def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index.
    
    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    :return: List of similar results
    """
    # Perform the similarity search using the FAISS index
    results = faiss_index.similarity_search(query, k=k)
    return results

# Sample YouTube URL
url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Fetching the transcript
transcript = get_transcript(url)

formatted_transcript = process(transcript)


chunks = chunk_transcript(processed_transcript)
# Output the chunks
print(chunks)
