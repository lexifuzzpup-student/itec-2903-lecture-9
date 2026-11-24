import os
from dotenv import load_dotenv
import requests
from langchain.tools import tool
from pprint import pprint

load_dotenv()
BASE_API_URL = os.environ["BASE_API_URL"]

plant_dict_content_instruction = """
Each plant has an integer id, name, scientific name, description, price, and care instructions.
Care instructions contain how much light is needed, how often to water, what soil to plant it in, what temperature ranges it thrives in, a tip for taking care of it, and how easy it is to take care of (care level).
"""

search_plants_by_price_tool_description = f"""
Takes a min_price and/or max_price in a dict.
Returns a list of plants within that price range.
{plant_dict_content_instruction}
Will return a string if an error occurs in the API.
"""

@tool(description = search_plants_by_price_tool_description)
def search_plants_by_price(query: dict) -> list[dict] | str:
    # create api url
    url = f"{BASE_API_URL}/search/price"

    # make web request
    response = requests.get(url, query)

    # error handling / response
    if response.status_code >= 300:
        return "Error calling plant price search API"
    else:
        return response.json()

search_plants_by_name_tool_description = f"""
Takes a plant name.
Returns a list of plants which match that name.
{plant_dict_content_instruction}
Makes case-insensitive, partial matches. For example, "bird" will match "Bird of Paradise."
Will return a string if an error occurs in the API.
"""

@tool(description = search_plants_by_name_tool_description)
def search_plants_by_name(plant_name: str) -> list[dict] | str:
    # create api url
    url = f"{BASE_API_URL}/search/name"

    # make web request
    response = requests.get(url, { "name": plant_name })

    # error handling / response
    if response.status_code >= 300:
        return "Error calling plant price search API"
    else:
        return response.json()

# if __name__ == "__main__":
    # pprint(search_plants_by_price({ "min_price": 5, "max_price": 15 }))
    # pprint(search_plants_by_name("bird"))