import os
import streamlit as st
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import GraphCypherQAChain

# Load environment variables from .env file
load_dotenv()

# Set Neo4j credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Set Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize the Neo4j graph
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

# Load the LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Create the Graph Cypher QA Chain
chain = GraphCypherQAChain.from_llm(graph=graph, llm=llm, verbose=True)

# Set up the Streamlit page
st.set_page_config(page_title="Movie Knowledge Graph", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: white;
            font-size: 20px;
        }
        .navbar {
            background-color: white;
            padding: 15px;
            text-align: center;
            border-bottom: 2px solid orange;
            margin-bottom: 25px;
        }
        .navbar h1 {
            color: orange;
            font-size: 2.5rem;
            margin: 0;
        }
        .stTextInput > div > input {
            font-size: 1.5rem;
            padding: 10px;
            border-radius: 10px;
            border: 2px solid orange;
        }
        .stButton > button {
            background-color: white;
            color: orange;
            border: 2px solid orange;
            border-radius: 10px;
            padding: 15px 30px;
            font-size: 1.5rem;
            transition: background-color 0.3s ease, color 0.3s ease;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: orange;
            color: white;
        }
        .stCheckbox > div > input {
            width: 25px;
            height: 25px;
        }
        .stCheckbox > label {
            font-size: 1.5rem;
        }
        .response-container {
            background-color: white;
            color: orange;
            padding: 20px;
            border-radius: 10px;
            font-size: 1.5rem;
            margin-top: 20px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
    <div class="navbar">
        <h1>Movie Knowledge Graph</h1>
    </div>
""", unsafe_allow_html=True)

# Input field for the user's query
user_query = st.text_input("Enter your question about movies:")

# Run the query when the user submits
if st.button("Ask"):
    if user_query:
        with st.spinner("Processing..."):
            response = chain.invoke({"query": user_query})
            
            # Inspect and format the response
            st.write(response)  # Temporarily show the full response structure for debugging
            
            # Assuming the 'answer' is under a key named 'result'
            answer = response.get('result', 'No answer found.')
            
            # Format the answer
            formatted_response = f"**Answer:** {answer}"
            st.markdown(f"<div class='response-container'>{formatted_response}</div>", unsafe_allow_html=True)

# Optionally, display the graph schema
if st.checkbox("Show Graph Schema"):
    graph.refresh_schema()
    st.write(graph.schema)
