import streamlit as st
import requests

# Backend API URL
API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(
    page_title="JW Infotech Multimodal RAG",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 JW Infotech Multimodal RAG")
st.write("Ask questions from documents, manuals, or images.")

# Input box
question = st.text_input(
    "Enter your question:",
    placeholder="e.g. Explain the wiring diagram"
)

# Ask button
if st.button("Ask AI"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": question},
                    timeout=60
                )

                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.success("Answer:")
                    st.write(answer)
                else:
                    st.error(f"Backend error: {response.status_code}")

            except Exception as e:
                st.error(f"Connection failed: {e}")
