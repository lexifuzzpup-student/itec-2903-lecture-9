import os
from dotenv import load_dotenv
import requests
from langchain.tools import tool

load_dotenv()
BASE_API_URL = os.environ["BASE_API_URL"]

order_status_tool_description = """
Returns the status of an order based on its ID.
An order consists of its order number (xxxxxxxxxx), order date (YYYY-MM-DD), current order status (processing, shipped, delivered, returned), and items in the order.
If the current order status is delivered or returned, the delivery date (YYYY-MM-DD) will be included.
Each item contains the product name, price of the product, and quantity ordered.
If an error occurs, a dictionary with one key, error, will be returned.
"""

# get the order status for an order id from the api
@tool(description=order_status_tool_description)
def get_order_status(order_id: int) -> dict:
    # create endpoint (there is a bug here, where order_id can be less than 1000000000, which produces a non-10-digit order id)
    url = f"{BASE_API_URL}/orders/status/{str(order_id)}"

    # make web request
    response = requests.get(url)

    # error handling / response
    if response.status_code >= 300:
        return { "error": "Error calling status API" }
    else:
        return response.json()


# errors in modern langchain

# if __name__ == "__main__":
#     order_status = get_order_status(8284241285)
    # print(order_status)