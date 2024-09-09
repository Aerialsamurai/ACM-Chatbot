from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.schema import HumanMessage, SystemMessage


load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


st.title('Langchain Chatbot by Aryan')


if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = [
        SystemMessage(content="Hi I am here to help you 😊")
    ]


model_choice = st.selectbox(
    "Choose a model to interact with:",
    ("LLaMA2", "Mistral-Nemo", "Gemma")
)


if model_choice == "LLaMA2":
    llm = Ollama(model="llama2")
    model_name = "LLaMA2"
elif model_choice == "Mistral-Nemo":
    llm = Ollama(model="mistral-nemo")
    model_name = "Mistral-Nemo"
else:
    llm = Ollama(model="gemma")
    model_name = "Gemma"


input_text = st.text_input("Ask me anything")


if input_text:
    
    st.session_state['chat_history'].append(HumanMessage(content=input_text))

    
    prompt = ChatPromptTemplate.from_messages(
        st.session_state['chat_history']
    )

  
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser


    response = chain.invoke({'question': input_text})

    st.session_state['chat_history'].append(SystemMessage(content=response))


for chat in st.session_state['chat_history']:
    if isinstance(chat, HumanMessage):
        st.write(f"**You**: {chat.content}")
    else:
        st.write(f"**{model_name}**: {chat.content}")
