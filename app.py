import streamlit as st
import os
import requests
import pickle
import time
from dotenv import load_dotenv
from streamlit_extras.badges import badge
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

# Constants
FILE_EXPIRATION_TIME = 7200  # 120 minutes in seconds

# Sidebar contents
with st.sidebar:
    st.title('🔗💬 LLM PDF Chat App')
    # st.markdown('''
    # ## About
    # This app is an LLM-powered pdf query bot built using :
    # - [Streamlit](https://streamlit.io/)
    # - [LangChain](https://python.langchain.com/)
    # - [Groq LLM Model](https://console.groq.com/docs/models) 
 
    # ''')
    add_vertical_space()
    st.write('Made with ❤️ by Anurag Shukla')
    badge(type='github',name='anuragshukla07',url='https://github.com/anuragshukla07')
    badge(type='twitter',name='_anuragshukla_',url='https://twitter.com/_anuragshukla_')
    # badge(type='buymeacoffee',name='anuragshukla07',url='https://linkedin.com/in/anuragshukla07')
    add_vertical_space()
    st.write('Please Provide Your Valuable Feedback :')
    st.write('[Feedback](https://docs.google.com/forms/d/e/1FAIpQLSdElFrQ7l04vFQzAoe3XIyju597pHFKSKohgJ6t66sZinss5g/viewform)')

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Function to clean up old files
def cleanup_old_files(directory=".", expiration_time=FILE_EXPIRATION_TIME):
    current_time = time.time()

    # Iterate through all files in the current directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Ensure we only delete .pkl files
        if filename.endswith(".pkl"):
            file_mod_time = os.path.getmtime(file_path)
            file_age = current_time - file_mod_time

            # Check if the file is older than the expiration time
            if file_age > expiration_time:
                os.remove(file_path)
                # st.write(f"Deleted old file: {filename} (Age: {file_age // 60} minutes)")
                
# Main logic
def main():
    st.title('Ask PDF')

    # Track state of the uploaded PDF and its associated data in session_state
    if 'uploaded_pdf' not in st.session_state:
        st.session_state.uploaded_pdf = None
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = None

    # Call the cleanup function to remove old .pkl files
    cleanup_old_files(directory=".", expiration_time=FILE_EXPIRATION_TIME)

    # PDF file uploader
    pdf = st.file_uploader('Upload Your PDF', type='pdf')

    # Handle if the file is uploaded
    if pdf is not None:
        st.session_state.uploaded_pdf = pdf  # Store in session state


        # Read the PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Split the text into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_text(text=text)

        # Create embeddings and FAISS vector store
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                st.session_state.vectorstore = pickle.load(f)
        else:
            with st.spinner("Uploading File Into The System..."):
                embeddings = HuggingFaceEmbeddings()
                st.session_state.vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(st.session_state.vectorstore, f)

    # If file is removed from the uploader
    if pdf is None and st.session_state.uploaded_pdf is not None:
        store_name = st.session_state.uploaded_pdf.name[:-4]
        # Remove the .pkl file if it exists
        if os.path.exists(f"{store_name}.pkl"):
            os.remove(f"{store_name}.pkl")
            # st.write(f"Deleted file {store_name} after PDF removal.")
        st.session_state.uploaded_pdf = None  # Clear the stored state

    # Process query input
    if st.session_state.uploaded_pdf is not None:
        query = st.text_input('Ask Queries To Your PDF File:')
        max_retries = 5
        retry_delay = 1
        if query:
            with st.spinner("Generating Results..."):
                for attempt in range(max_retries):
                    try:
                        docs = st.session_state.vectorstore.similarity_search(query)
                        llm = ChatGroq(groq_api_key=groq_api_key,model_name = 'llama-3.1-70b-versatile' , timeout = 60)
                        chain = load_qa_chain(llm,chain_type='stuff')
                        response = chain.run(input_documents=docs,question=query)
                        time.sleep(3)
                        break
                    except requests.exceptions.HTTPError as err:
                        if(err.response.status_code == 503): # Handling Server Unavailable
                            st.warning(f"Server Unavailable , Retrying In {retry_delay} Seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2
                        elif (err.response.status_code == 429):  # Handling rate limiting (Too Many Requests)
                            st.warning("Too many requests. Please wait and try again later.")
                            time.sleep(retry_delay * 5)    
                        else:
                            st.error(f"An HTTP Error Occurred : {err}")
                            raise
                    except requests.exceptions.RequestException as err:
                        st.error(f"An Error Occurred: {err}")
                        raise  # Handle other types of request-related errors
                    except Exception as e:
                        st.error(f"An Unexpected Error Occurred: {e}")
                        raise  # Handle any other exceptions

            st.write(response)

if __name__ == '__main__':
    main()
