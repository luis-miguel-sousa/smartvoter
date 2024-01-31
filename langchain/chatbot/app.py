
import streamlit as st
from streamlit_chat import message
from llama import *

st.set_page_config(page_title='Llama2-Chatbot')
st.header('Custom Llama2-Powered Chatbot :robot_face:')


def get_user_input():

    # get the user query
    input_text = st.text_input('Ask me anything about the portuguese elections', "", key='input')
    return input_text

# create the qa_chain
qa_chain = create_qa_chain()

# create empty lists for user queries and responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []

# get the user query
user_input = get_user_input()

if user_input:

    # generate response to the user input
    response = generate_response(query=user_input, qa_chain=qa_chain)

    # add the input and response to session state
    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

# display conversaion history (if there is one)
if st.session_state['generated']:
    for i in range(len(st.session_state['generated']) -1, -1, -1):
        message(st.session_state['generated'][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')