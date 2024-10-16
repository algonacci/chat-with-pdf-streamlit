import os
import streamlit as st
from helpers import *
import time


def main():
    st.title("RAG Chat Application")

    uploaded_file = st.file_uploader("Choose a file for context", type=["txt", "pdf"])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            process_document(uploaded_file)
        st.success(f"Document processed: {uploaded_file.name}")
    else:
        st.info("Please upload a document to provide context for the chat.")


    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            thinking_placeholder = st.empty()
            thinking_text = "Thinking"
            
            for _ in range(3):
                for dots in range(4):
                    thinking_placeholder.markdown(f"**{thinking_text}{'.' * dots}**")
                    time.sleep(0.1)

            thinking_placeholder.empty()

            full_response = ""

            for chunk in rag_response(prompt):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")

            message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})



if __name__ == "__main__":
    main()