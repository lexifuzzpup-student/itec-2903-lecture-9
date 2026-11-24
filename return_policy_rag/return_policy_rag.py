import json
import os
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
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

# return_policy_document.md in the directory of this file
return_policy_document = os.path.join(directory, "return_policy_document.md")

return_rag_system_prompt = """
Answer the customer question using only the information provided in the context.
If the context doesn't provide enough information to answer the question, direct the user to our customer support team.
If the customer asks about anything other than refund information found in the documents, reply with "I can't answer that."
No hallucinating, be truthful, and don't create new information.
"""

return_rag_tool_description = """
Reads a question about plant returns, returning knowledge from our plant return policy documents.
If the customer's question is unrelated, "I can't answer that" will be returned.
If the customer's question cannot be answered properly, it will direct the customer to the customer support team.
"""

def create_retriever(generate_embeddings = False):
    gemini_embeddings = GoogleGenerativeAIEmbeddings(model = EMBEDDING_MODEL)
    collection_name = "return_policy_documents"
    persist_directory = os.path.join(directory, "return_policy_embeddings")

    if generate_embeddings:
        # create new document embeddings using the model. markdown file is split by its headings, turning it into separate documents.
        with open(return_policy_document, "r") as file:
            policy_document = file.read()

        # declare what headers to split on
        headers_to_split_on = [
            ( "#", "Header 1" ),
            ( "##", "Header 2" )
        ]

        # split document by headers
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
        documents = markdown_splitter.split_text(policy_document)

        # create new document embeddings with chroma
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

    # query = "One of the plants was broken when it arrived"
    # results = vector_store.similarity_search(query, k = 3)
    # for result in results: print(result)

    # "mmr" finds a wide variety of documents that match a prompt
    # "similarity" finds the most similar documents that match an input
    retriever = vector_store.as_retriever(
        search_type = "similarity",
        search_kwargs = { "k": 3 } # return 3 documents that best match
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
        input_variables = [ "instructions", "context", "question" ]
    )

    # create langchain wrapper for gemini
    llm = ChatGoogleGenerativeAI(model = RAG_LLM_MODEL, temperature = 0.2)

    plant_return_rag_chain = (
        {
            "instructions": lambda _: return_rag_system_prompt,
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return plant_return_rag_chain


retriever = create_retriever(False)
return_rag_chain = create_rag_chain(retriever)

@tool(description = return_rag_tool_description)
def return_policy_rag_query(query: str) -> str:
    return return_rag_chain.invoke(query)

if __name__ == "__main__":
    create_retriever(True)