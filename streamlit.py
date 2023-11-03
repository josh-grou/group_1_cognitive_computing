import streamlit as st
from streamlit_chat import message
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
import tempfile
import json

f = open("apikey.json")
data = json.load(f)
key = data["openaikey"]

embeddings = OpenAIEmbeddings(openai_api_key=key)

db = FAISS.load_local("faiss_dump",embeddings)
chain = ConversationalRetrievalChain.from_llm(
llm = ChatOpenAI(temperature=0.15,model_name='gpt-3.5-turbo', openai_api_key=key),
retriever=db.as_retriever())

#container for the chat history
response_container = st.container()
#container for the user's text input
container = st.container()

def conversational_chat(query):
    
    result = chain({"question": query, 
    "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    
    return result["answer"]

if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'generated' not in st.session_state:
    st.session_state['generated'] = ["Hello! I am your Food.com personal chef, ask me anything and I can use over 230 thousand recipes to help you make a meal! 🤗"]

if 'past' not in st.session_state:
    st.session_state['past'] = ["Hey ! 👋"]
    
with container:
    with st.form(key='my_form', clear_on_submit=True):
        
        user_input = st.text_input("Query:", placeholder="Talk to me about your cooking problems here!", key='input')
        submit_button = st.form_submit_button(label='Send')
        
    if submit_button and user_input:
        output = conversational_chat(user_input)
        
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")