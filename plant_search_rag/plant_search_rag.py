import json
import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

load_dotenv()
EMBEDDING_MODEL = os.environ["EMBEDDING_MODEL"]
RAG_LLM_MODEL = os.environ["RAG_LLM_MODEL"]

# current directory this exact file is in
directory = Path(__file__).resolve().parent

# plants.json in the directory of this file
plant_database_file = os.path.join(directory, "plants.json")

plant_rag_system_prompt = """
Answer the customer question using only the information provided in the context.
If the context doesn't provide enough information to answer the question, reply with "I don't know."
If the customer asks about anything other than house plants, remind them that you're only able to find and present applicable information regarding house plants.
"""

plant_rag_tool_description = """
Reads a question about house plants, returning knowledge from within the database.
Questions can give a summary of plant care requirements, light, humidity, soil, descriptions, and tips.
If the answer can't be provided, "I don't know" will be returned.
"""

def plant_json_to_text(plant: dict) -> str:
    return f"""
    Plant name: {plant['name']} 
    Description: {plant['description']} 
    Scientific name: {plant['scientific_name']}
    Care, light levels: {plant['care']['light']}
    Care, water needs: {plant['care']['water']}
    Care, soil: {plant['care']['soil']}
    Care, prefered temperature and humidity: {plant['care']['temperature_and_humidity']}
    Care, tips: {plant['care']['tips']}
    """

def create_retriever(generate_embeddings = False):
    collection_name = "plant_details"
    persist_directory = os.path.join(directory, "plant_embeddings")

    gemini_embeddings = GoogleGenerativeAIEmbeddings(model = EMBEDDING_MODEL)

    if generate_embeddings:
        # create new document embeddings using the embedding model
        data = json.load(open(plant_database_file))

        # convert json entries for plants into human-readable representations of the same data
        texts = [ plant_json_to_text(plant) for plant in data ]

        # convert raw document text into document instances
        documents = [ Document(page_content = text) for text in texts ]

        vector_store = Chroma.from_documents(
            collection_name = collection_name,
            documents = documents,
            embedding = gemini_embeddings,
            persist_directory = persist_directory
        )
    else:
        # use existing document embeddings from chroma
        vector_store = Chroma(
            collection_name = collection_name,
            embedding_function = gemini_embeddings,
            persist_directory = persist_directory
        )

    # "mmr" finds a wide variety of documents that match a prompt
    # "similarity" finds the most similar documents that match an input
    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = { "k": 5, "fetch_k": 15 } # fetch 15 documents, pick 5 that answer the question best
    )

    return retriever


def create_rag_chain(retriever):
    rag_prompt = PromptTemplate(
        template = """
            {instructions}
            
            Use this context to answer the question:

            {context}

            Customer question:

            {question}

            Response:
        """,
        input_variables = [ "instructions", "question", "context" ]
    )

    # create langchain wrapper for gemini
    llm = ChatGoogleGenerativeAI(model = RAG_LLM_MODEL, temperature = 0.5)

    plant_rag_chain = (
        {
            "instructions": lambda _: plant_rag_system_prompt,
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return plant_rag_chain

retriever = create_retriever(False)
plant_rag_chain = create_rag_chain(retriever)

@tool(description = plant_rag_tool_description)
def plant_rag_query(query: str) -> str:
    return plant_rag_chain.invoke(query)

if __name__ == "__main__":
    create_retriever(True)

    # query = "What is a good plant for a bathroom with high humidity?"
    # rag_result = plant_rag_chain.invoke(query)
    # print(rag_result)
