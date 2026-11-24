import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from tools.order_status_tools import get_order_status

# read .env
load_dotenv()
LLM_MODEL = os.environ["AGENT_LLM_MODEL"]

# system prompts (i think drier language signifies better that this is a tool, not a conversation)
agent_system_prompt = """
You are a human customer assistant for a house plant store.
Do not use emojis, do not include introductions, and do not praise the user. Do not read responses verbatim.
Only provide information that exists within our database.
Include extra returned information if it's applicable to the request, since the user will likely want to know every detail regarding their request.
Use friendly, non-technical, understandable language and keep responses short.
Use natural language (contractions, phrases like "I don't know," and "I can't," etc.).
If the user asks for assistance about anything that you can't access about our store, politely refuse to answer, reminding them that you're only able to find and present applicable information regarding our store.
"""

# create front-facing agent
agent = create_agent(
    model = LLM_MODEL,
    tools = [ get_order_status ],
    system_prompt = agent_system_prompt,

    # save conversations in memory, referenced by a thread id
    checkpointer = InMemorySaver()
)

# create a text respones for an input message
def generate_response(user_message: str, thread_id: str) -> str:
    human_message = HumanMessage(user_message)

    # query the llm
    response = agent.invoke({
        "messages": [ human_message ]
    }, {
        # the thread is is used to distinguish conversation contexts
        "configurable": {
            "thread_id": thread_id
        }
    })

    messages = response["messages"]

    # get last message (the llm response)
    ai_message = messages[-1]
    ai_message_text = ai_message.text

    return ai_message_text

if __name__ == "__main__":
    thread_id = "testing"
    generate_response(input("Enter query - "), thread_id)
