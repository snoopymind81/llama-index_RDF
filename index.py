import os, streamlit as st

# Uncomment to specify your OpenAI API key here (local testing only, not in production!), or add corresponding environment variable (recommended)
# os.environ['OPENAI_API_KEY']= ""


from pathlib import Path
from llama_index import GPTSimpleVectorIndex, LLMPredictor, PromptHelper, download_loader, Document
from langchain import OpenAI

# This example uses text-davinci-003 by default; feel free to change if desired
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

# Configure prompt parameters and initialise helper
max_input_size = 4000
num_output = 256
max_chunk_overlap = 20

prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# Load documents from the 'data' directory
# documents = SimpleDirectoryReader('data').load_data()

#JSONReader = download_loader("JSONReader")

#loader = JSONReader()
#documents = loader.load_data(file=Path('./LibraryMaterial_json_20230320.json'))

# RDF
RDFReader = download_loader("RDFReader")

loader = RDFReader()
documents = loader.load_data(file=Path('./ot_result_ver02.ttl'))


index = GPTSimpleVectorIndex(
    documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
)

# Define a simple Streamlit app
st.title("도서관 정보 온톨로지")
query = st.text_input("What would you like to ask?", "")

if st.button("Submit"):
    response = index.query(query)
    st.write(response)
