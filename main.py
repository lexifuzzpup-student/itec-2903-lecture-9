from agent import generate_response
import rich
from rich.markdown import Markdown
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

def agent_chat():
    print("Welcome to the Plant Store support tool. Ask a question regarding the store. Requests are processed by AI.")

    thread_id = uuid4()
    print(thread_id)

    while True:
        user_message = input("> ")

        if not user_message:
            print("Goodbye!")
            break

        ai_message = generate_response(user_message, str(thread_id))
        rich.print(Markdown(ai_message))

if __name__ == "__main__":
    agent_chat()