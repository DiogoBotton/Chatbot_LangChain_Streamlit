import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

from enums import ClassType

load_dotenv()

class_types = {ClassType.OPENAI: "OpenAI API", ClassType.OLLAMA: "Llama 3 com Ollama (execução local)"}

def format_func(opt):
    return class_types[opt]

model_class_type = st.selectbox("Escolha o modelo", options=class_types.keys(), format_func=format_func)

# Configurações do Streamlit
st.set_page_config(page_title="Seu assistente virtual 🤖", page_icon="🤖")
st.title("Seu assistente virtual 🤖")

def model_openai(model_name = "gpt-4o-mini", temperature = 0.1):
    """
    Acessa o modelo do Chat GPT pela API.
    """
    llm = ChatOpenAI(model = model_name, temperature = temperature)
    return llm

def model_ollama(model = "llama3.1:8b", temperature = 0.1):
    """
    Utiliza um modelo instalado localmente com Ollama.
    """
    llm = ChatOllama(model=model, temperature=temperature)
    return llm

def model_response(user_query, chat_history, model_class):
    # Carregamento da LLM
    match model_class:
        case ClassType.OPENAI:
            llm = model_openai()
        case ClassType.OLLAMA:
            llm = model_ollama()
        case _: # default
            raise Exception("Tipo de modelo não é válido.")
        
    # Definição dos prompts
    system_prompt = "Você é um assistente prestativo e está respondendo perguntas gerais. Responda em {language}."
    language = "português"
    
    # Definição do prompt template de Chat
    # O MessagesPlaceHolder é usado para armazenar o histórico das conversas
    prompt_template = ChatPromptTemplate([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    # Criação da Chain
    chain = prompt_template | llm | StrOutputParser()
    
    # stream retornará as respostas dinamicamente enquanto são geradas, invoke retornará tudo de uma vez
    return chain.stream({"chat_history": chat_history,
                         "input": user_query,
                         "language": language})

# Cria variável chat_history caso não exista
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Olá, sou o seu assistente virtual! Como posso ajudar você? :)")]

# Renderização do histórico de mensagens
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("ai"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("human"):
            st.write(message.content)

# Input para o usuário escrever            
user_query = st.chat_input("Digite sua mensagem aqui...")

# Adiciona o input ao chat
if user_query is not None and len(user_query) != 0:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("human"):
        st.markdown(user_query)
    
    with st.chat_message("ai"):
        resp = st.write_stream(model_response(user_query,
                                              st.session_state.chat_history,
                                              model_class_type))
    
    st.session_state.chat_history.append(AIMessage(content=resp))