import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.messages import HumanMessage
from tools.order_status_tools import get_order_status

# read .env
load_dotenv()
LLM_MODEL = os.environ["AGENT_LLM_MODEL"]

# system prompts (i think drier language signifies better that this is a tool, not a conversation)
agent_system_prompt = """
You are an customer assistant for a house plant store. 
Do not use emojis, do not include introductions, and do not praise the user, but do not read instructions verbatim.
Only provide information that exists within our database.
Include extra returned information if it's applicable to the request, since the user will likely want to know every detail regarding their request.
Use friendly, non-technical language and keep responses short.
If the user asks for assistance about anything that our resources don't cover, politely refuse to answer, reminding them that your purpose is to find and present applicable information regarding our store.
"""

# create front-facing agent
agent = create_agent(
    model = LLM_MODEL,
    tools = [ get_order_status ],
    system_prompt = agent_system_prompt
)

# create a text respones for an input message
def generate_response(user_message: str):
    human_message = HumanMessage(user_message)

    # query the llm
    response = agent.invoke({
        "messages": [ human_message ]
    })

    messages = response["messages"]

    for message in messages:
        print(message.pretty_repr())

    # get last message (the llm response)
    ai_message = messages[-1]
    ai_message_text = ai_message.text

    print(ai_message_text)

if __name__ == "__main__":
    generate_response(input("Enter query - "))
