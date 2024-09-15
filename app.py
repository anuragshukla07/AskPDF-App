import streamlit as st
import requests
import time
import pickle
import os
from streamlit_extras.badges import badge
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq


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
def main():
    st.title('Ask PDF')
    pdf = st.file_uploader('Upload Your PDF',type='pdf')
    if(pdf is not None):
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter =CharacterTextSplitter(
        # Set a really small chunk size, because the LLMs have limited number of tokens
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )

        chunks = text_splitter.split_text(text=text)

        #embeddings
        store_name = pdf.name[:-4]
        if os.path.exists(f"{store_name}.pkl"): 
            with open(f"{store_name}.pkl","rb") as f:
                VectorStore = pickle.load(f)
        else:
            with st.spinner("Uploading File Into The System..."):
                embeddings = HuggingFaceEmbeddings()
                VectorStore = FAISS.from_texts(chunks,embedding=embeddings)
                with open(f"{store_name}.pkl","wb") as f:
                    pickle.dump(VectorStore,f)
 
        query = st.text_input('Ask Queries To Your PDF File :') 

        max_retries = 5
        retry_delay = 1

        if query:
            with st.spinner("Generating Results..."):
                for attempt in range(max_retries):
                    try:
                        docs = VectorStore.similarity_search(query)
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
            # st.toast('Data Fetched Successfully!')
            # st.success('Data Fetched Successfully!')


if __name__ == '__main__':
    main()