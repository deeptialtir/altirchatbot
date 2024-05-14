import streamlit as st
import random
import time
from hugchat import hugchat
from hugchat.login import Login
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

# Load the LlamaCpp language model, adjust GPU usage based on your hardware
llm = LlamaCpp(model_path="models/llama-2-7b-chat.Q5_K_M.gguf", n_gpu_layers=1, n_batch=10, verbose=False)  

# Streamed response emulator
def response_generator(user_input):
    template = """
        Question: {user_input}

        Answer:
        """
    prompt = PromptTemplate(template=template, input_variables=["user_input"])

    # Create an LLMChain to manage interactions with the prompt and model
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    print("Chatbot initialized, ready to chat...")
    
    question = user_input
    print("Question is %s" %(question))
    answer = llm_chain.run(question)
    print(answer, '\n')
    response = answer

    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# Hugchat Headings
st.title("Altir: SpaceX ChatBot Demo")

with st.sidebar:
    st.title('🅰️💬 Altir ChatBot Demo')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](<https://streamlit.io/>)
    - [HugChat](<https://github.com/Soulter/hugging-chat-api>)
    - [OpenAssistant/oasst-sft-6-llama-30b-xor](<https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor>) LLM model

    💡 Note: No API key required!
    ''')
    add_vertical_space(5)
    

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(user_input=prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
