import os
from flask import Flask, render_template, request, jsonify, redirect
import PyPDF2
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma  # type: ignore
from dotenv import load_dotenv
import anthropic
from flask_cors import CORS  
from pymongo import MongoClient
import requests

app = Flask(__name__)
CORS(app)  
load_dotenv()

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client.os_chatbot
user_questions_collection = db.user_questions

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Anthropic API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("Missing Anthropic API key.")
client_anthropic = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Load IPinfo and OpenCage API keys from .env
IPINFO_TOKEN = os.getenv("IPINFO_TOKEN")
OPENCAGE_API_KEY = os.getenv("OPENCAGE_API_KEY")
if not IPINFO_TOKEN or not OPENCAGE_API_KEY:
    raise ValueError("Missing IPinfo or OpenCage API key.")

vectorstore = None
#FRONT_END_BASE_URL = os.getenv("FRONT_END_BASE_URL", "http://127.0.0.1:5500")

def process_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def split_text(text, chunk_size=1000, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)

def perform_similarity_search(query, k=1):
    global vectorstore
    if vectorstore is None:
        return []
    return vectorstore.similarity_search(query, k=k)

def generate_response(context, query):
    prompt = f"""
You are an AI assistant with extensive knowledge and reasoning capabilities.
Answer the user's question using both the retrieved context and your own general knowledge.

- Use the context when it is relevant, but do not rely solely on it.
- Provide a structured, insightful, and easy-to-understand response.
- **Return your answer strictly in valid HTML**.
- Use headings (<h2>, <h3>), paragraphs (<p>), bold text, lists (<ul>, <li>), and so on.
-Please return only plain HTML and do not use code fences like ```html or any Markdown code blocks.
- 

### Context (if relevant):
{context}

### Question:
{query}

### Answer in HTML:
"""
    try:
        message = client_anthropic.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def initialize_vectorstore():
    global vectorstore
    file_path = "notes.pdf"
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return False
    try:
        pdf_text = process_pdf(file_path)
        chunks = split_text(pdf_text)
        vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name="dbms_notes_chunks"
        )
        return True
    except Exception as e:
        print(f"Error initializing vector store: {str(e)}")
        return False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    try:
        data = request.json
        user_query = data.get('question', '')

        # Step 1: Capture the user's IP address
        user_ip = request.headers.get('X-Forwarded-For', request.remote_addr)

        # Step 2: Fetch latitude and longitude from IPinfo.io
        location_info = {}
        if user_ip:
            try:
                ipinfo_response = requests.get(f"https://ipinfo.io/{user_ip}/json?token={IPINFO_TOKEN}")
                ipinfo_data = ipinfo_response.json()
                if "loc" in ipinfo_data:
                    lat, lon = ipinfo_data["loc"].split(",")
                    location_info = {
                        "latitude": float(lat),
                        "longitude": float(lon)
                    }
            except Exception as e:
                print(f"Error fetching IPinfo data for {user_ip}: {str(e)}")

        # Step 3: Reverse geocode with OpenCage for exact location
        exact_location = None
        if location_info.get("latitude") and location_info.get("longitude"):
            try:
                opencage_url = f"https://api.opencagedata.com/geocode/v1/json?q={location_info['latitude']}+{location_info['longitude']}&key={OPENCAGE_API_KEY}"
                reverse_response = requests.get(opencage_url)
                reverse_data = reverse_response.json()
                if reverse_data.get("results") and len(reverse_data["results"]) > 0:
                    exact_location = reverse_data["results"][0].get("formatted")
            except Exception as e:
                print(f"Error reverse geocoding with OpenCage: {str(e)}")

        # Step 4: Save everything to MongoDB
        if user_query:
            question_doc = {
                "question": user_query,
                "type": "user_query",
                "ip_address": user_ip
            }
            if location_info:
                question_doc["location"] = {
                    "latitude": location_info["latitude"],
                    "longitude": location_info["longitude"]
                }
            if exact_location:
                question_doc["exact_location"] = exact_location
            user_questions_collection.insert_one(question_doc)

        if not user_query:
            return jsonify({'error': 'No question provided'}), 400

        global vectorstore
        if vectorstore is None:
            if not initialize_vectorstore():
                return jsonify({'error': 'Failed to initialize vector store'}), 500

        results = perform_similarity_search(user_query, k=2)
        if not results:
            return jsonify({'error': 'No relevant context found'}), 404

        context = "\n".join([doc.page_content for doc in results])
        response = generate_response(context, user_query)

        return jsonify({
            'answer': response,
            'context': context
        })
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if initialize_vectorstore():
        port = int(os.getenv("PORT", 10000))
        app.run(host="0.0.0.0", port=port, debug=True)
    else:
        print("Failed to initialize vector store. Please check the PDF file and try again.")