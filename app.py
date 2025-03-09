import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import faiss
### from langchain.memory import ConversationBufferMemory # See UPDATE.
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
import streamlit as st
import nest_asyncio


nest_asyncio.apply()
load_dotenv()


# Initialize app resources
st.set_page_config(page_title="Simple Electricity Bill Reader", page_icon=":book:")
st.title("Simple Electricity Bill Reader (v0)")
st.write("An AI-RAG application to assist reading electricity bill into JSON.")


@st.cache_resource
def initialize_resources():
    llm_gemini = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest", google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return llm_gemini


def get_retriever(pdf_file):
    # with NamedTemporaryFile(suffix="pdf") as temp: # Got the error "Permission denied".
    # current_app_path = os.path.dirname(os.path.abspath(__file__))
    # local_temp_path = os.path.join(current_app_path, 'temp')
    local_temp_path = os.path.join(os.getcwd(), 'temp')
    os.makedirs(local_temp_path, exist_ok=True)  # Ensure the temp directory exists
    with NamedTemporaryFile(suffix=".pdf", dir=local_temp_path, delete=False) as temp:
        temp.write(pdf_file.getvalue())
        pdf_loader = PyPDFLoader(temp.name, extract_images=True)
        pages = pdf_loader.load()

    underlying_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
        separators="\n",
    )
    documents = text_splitter.split_documents(pages)
    vectorstore = faiss.FAISS.from_documents(documents, underlying_embeddings)
    doc_retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 10}
    )

    return doc_retriever


chat_model = initialize_resources()


### start UPDATE

# LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
# LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
#   memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# from langchain.memory import ConversationSummaryMemory  # Updated memory class # requires `pip install langchain-core`
from langchain.memory import ConversationSummaryBufferMemory  # Updated memory class # requires `pip install langchain-core`

### end UPDATE


doc_retriever = None
conversational_chain = None
input_bill = "{Not selected}"


def query_response(query, _retriever):
    ### memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
    ### memory = ConversationSummaryMemory(llm=chat_model, memory_key="chat_history", return_messages=True)  # Updated memory implementation
    memory = ConversationSummaryBufferMemory(llm=chat_model, memory_key="chat_history", return_messages=True, max_token_limit=1000)  # Updated memory implementation
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model, retriever=_retriever, memory=memory, verbose=False
    )

    # Ensure chat_history is included in the input
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat history if not present

    ### response = conversational_chain.run(query) # LangChainDeprecationWarning: The method `Chain.run` was deprecated in langchain 0.1.0 and will be removed in 1.0. Use :meth:`~invoke` instead.
    response = conversational_chain.invoke({"question": query})

    answer = response.get("answer", "")

    # return response
    return answer


if "doc" not in st.session_state:
    st.session_state.doc = ""

input_bill = st.file_uploader("or Upload your own pdf", type="pdf")

if st.session_state != "":
    try:
        with st.spinner("loading document.."):
            doc_retriever = get_retriever(input_bill)
        st.success("File loading successful, vector db initialised.")
    except Exception as e:
        st.error(e)

    # We store the conversation in the session state.
    # This will be used to render the chat conversation.
    # We initialize it with the first message we want to be greeted with.
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I assist you today?"}
        ]

    if "current_response" not in st.session_state:
        st.session_state.current_response = ""

    # We loop through each message in the session state and render it as
    # a chat message.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # We take questions/instructions from the chat input to pass to the LLM
    ### if user_prompt := st.chat_input("Ask...", key="user_input"):
    if user_prompt := st.chat_input("Ask..."):
        # Add our input to the session state
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        # Add our input to the chat window
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Pass our input to the llm chain and capture the final responses.
        # here once the llm has finished generating the complete response.
        response = query_response(user_prompt, doc_retriever)

        # Add the response to the session state
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Add the response to the chat window
        with st.chat_message("assistant"):
            st.markdown(response)
