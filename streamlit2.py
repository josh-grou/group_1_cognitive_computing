import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
import json
import os

# Load the API key
with open("apikey.json") as f:
    data = json.load(f)
    key = data["openaikey"]

# Set up embeddings and the vector store
embeddings = OpenAIEmbeddings(openai_api_key=key)

# Load the vector store
faiss_index_path = os.path.join(os.getcwd(), "faiss_dump")
db = FAISS.load_local(faiss_index_path, embeddings)

# Initialize the conversational chain with the language model
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.9, model_name='gpt-3.5-turbo', openai_api_key=key),
    retriever=db.as_retriever())

# Initialize 'generated' and 'history' in session state if they don't exist
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'stage' not in st.session_state:
    st.session_state['stage'] = "greeting"

# Define the stages of the conversation
STAGES = {
    "greeting": "Hello! I'm Chef Bot, your culinary assistant. Let's start by narrowing down your preferences. What type of cuisine are you interested in?",
    "cuisine": "Great choice! Do you have any dietary restrictions?",
    "diet": "Understood. How much time do you have for cooking?",
    "time": "Got it. What are the main ingredients you'd like to use?",
    "ingredients": "Excellent. Let me find the best recipe for you based on your preferences."
}

# Helper function to determine the next stage
def next_stage_key(current_stage):
    keys = list(STAGES.keys())
    current_index = keys.index(current_stage)
    next_index = current_index + 1 if current_index + 1 < len(keys) else 0
    return keys[next_index]

# Helper function to format the recipe response
def format_recipe(recipe_text):
    formatted_recipe = "Recipe Details:\n" + recipe_text
    return formatted_recipe

# Define the conversation function with guided responses
def conversational_chat(query, stage):
    # Add the user query to the history
    st.session_state['history'].append({"user": query, "stage": stage})

    # Generate the response based on the current stage
    if stage == "ingredients":
        # Final stage: Find and format the recipe
        context = " ".join([exchange["user"] for exchange in st.session_state['history'] if "user" in exchange])
        inputs = {"question": query, "context": context, "chat_history": [exch["user"] for exch in st.session_state['history']]}
        response = chain.run(inputs)
        answer = format_recipe(response["answer"])
        st.session_state['stage'] = "end"  # End the conversation
    else:
        # Intermediate stages: Guide the conversation
        answer = STAGES[stage]
        st.session_state['stage'] = next_stage_key(stage)

    st.session_state['history'].append({"bot": answer, "stage": st.session_state['stage']})
    return answer

# Streamlit app UI
st.title('👩‍🍳 Your Personal Chef AI')

# Display the conversation based on the stage
st.write(STAGES[st.session_state['stage']])

# User input
user_input = st.text_input("Your response:", key='input')

# Handle the conversation based on the user input
if user_input:
    output = conversational_chat(user_input, st.session_state['stage'])
    st.session_state['generated'].append(output)

# Display the conversation history
with st.container():
    for i, exchange in enumerate(st.session_state['history']):
        if 'user' in exchange:
            message(exchange['user'], is_user=True, key=f"user_{i}")
        if 'bot' in exchange:
            message(exchange['bot'], key=f"bot_{i}")
