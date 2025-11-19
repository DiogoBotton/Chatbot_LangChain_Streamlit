import streamlit as st

# Importação de bibliotecas para a criação da LLM
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Importação de bibliotecas para realizar o RAG
import faiss # Banco de dados vetorial, busca de similaridade e agrupamento de vetores
import tempfile
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

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
    """
    Modelo com histórico, sem RAG.
    """
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

def config_retriever(uploads):
    # Carregar documentos
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploads:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        
        loader = PyPDFLoader(temp_filepath)
        docs.extend(loader.load())
    
    # Divisão em pedaços de texto / split
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, # Dividirá o texto em pedaços de 1000 caracteres
                                               chunk_overlap = 200,
                                               add_start_index = True) # Para o índice dos caracteres inicial seja preservado
    splits = text_splitter.split_documents(docs)
    
    # Embeddings
    ollama_embedding = OllamaEmbeddings(
        model="bge-m3:567m"
    )
    
    # Armazenamento
    vectorstore = FAISS.from_documents(splits, ollama_embedding)
    vectorstore.save_local("vectorstore/db_faiss")
    
    # Configuração do retriever
    
    # MMR - Seleciona os documentos baseado na relevância e diversidade entre os documentos recuperados, para evitar retornar contextos duplicados
    # Garante um melhor equilíbrio entre a relevância e a diversidade dos itens recuperados.
    
    # K -> Total de documentos recuperados
    # fetch_k -> Quantos documentos são recuperados antes de aplicar o algoritmo de MMR (valor padrão: 20)
    
    # Primeiro é adquirido os documentos (4), após isso, é aplicado o algoritmo de MMR para retornar os docs mais relevantes (3)
    retriever = vectorstore.as_retriever(search_type = "mmr", # Maximum Marginal Relevance Retriever
                                         search_kwargs={"k": 3, "fetch_k": 4})
    
    return retriever

def config_rag_chain(model_class, retriever):
    """
    Modelo com histórico e RAG.
    """
    # 1. Carregamento da LLM
    match model_class:
        case ClassType.OPENAI:
            llm = model_openai()
        case ClassType.OLLAMA:
            llm = model_ollama()
        case _: # default
            raise Exception("Tipo de modelo não é válido.")
    
    # 2. Prompt para reformular a pergunta baseado no histórico
    # Antes (na aula de RAG) -> consulta -> retriever
    # Agora (RAG com histórico) -> (consulta, histórico) -> LLM -> consulta reformulada baseado no histórico -> retriever
    context_q_system_prompt = "Given the following chat history and the follow-up question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    
    context_q_prompt = ChatPromptTemplate([
        ("system", context_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {input}")
    ])
    
    # 3. Chain para contextualização e reformular a pergunta
    rewrite_chain = context_q_prompt | llm | StrOutputParser()
    
    # 4. Pipeline de retrieval com LCEL
    # (input + histórico) -> pergunta reescrita -> retriever -> docs
    retrieval_chain = RunnableParallel({
        "input": RunnablePassthrough(),
        "rephrased_question": rewrite_chain
    }) | {
        "input": RunnablePassthrough(),
        "context": lambda x: retriever.invoke(x["rephrased_question"])
    }
    
    # 5. Prompt de resposta final voltado a perguntas e respostas (Q&A)
    qa_prompt_template = """Você é um assistente virtual prestativo e está respondendo perguntas gerais.
    Use os seguintes pedaços de contexto recuperado para responder à pergunta.
    Se você não sabe a resposta, apenas diga que não sabe. Mantenha a resposta concisa.
    Responda em português. \n\n
    Pergunta: {input} \n
    Contexto: {context}"""
    
    qa_prompt = PromptTemplate.from_template(qa_prompt_template)
    
    # 6. Configurar LLM e Chain para perguntas e respostas (Q&A)
    qa_chain = qa_prompt | llm | StrOutputParser()
    
    # 7. RAG final combinado
    rag_chain = retrieval_chain | qa_chain
    
    return rag_chain

def config_route_chain(model_class):
    """
    Fará a decisão se o modelo deve enviar para o modelo com RAG ou simplesmente para o modelo de conversação.    
    """
    match model_class:
        case ClassType.OPENAI:
            llm = model_openai()
        case ClassType.OLLAMA:
            llm = model_ollama()
        case _: # default
            raise Exception("Tipo de modelo não é válido.")
        
    route_prompt = ChatPromptTemplate.from_messages([
        ("system", 
            "Determine if the user message is asking about the documents or is casual conversation.\n"
            "Respond STRICTLY with one word: 'retrieval' or 'conversation'."),
        ("human", "{input}")
    ])
    
    route_chain = route_prompt | llm | StrOutputParser()
    
    return route_chain

def final_chain(input):
    """
    Envia o input do usuário para algum dos modelos dependendo da pergunta (modelos com RAG ou sem)
    """
    route_chain_llm = config_route_chain(model_class_type)
    route = route_chain_llm.invoke({"input": input})
    
    if "retrieval" in route.lower():
        pass # TODO: Definir resp aqui também
    else:
        resp = st.write_stream(model_response(user_query,
                                              st.session_state.chat_history,
                                              model_class_type))
    st.session_state.chat_history.append(AIMessage(content=resp))

uploads = st.sidebar.file_uploader(
    label="Enviar arquivos", type=["pdf"],
    accept_multiple_files=True
)

# Define variáveis de sessão
# Cria variáveisvel chat_history, docs_list e retriever caso não existam
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Olá, sou o seu assistente virtual! Como posso ajudar você? :)")]
    
if "docs_list" not in st.session_state:
    st.session_state.docs_list = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

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
        final_chain(user_query)