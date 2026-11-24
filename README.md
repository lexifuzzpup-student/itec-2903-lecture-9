# itec-2903-lecture-9

## Setup
Copy the `.env.example` file to `.env` and fill in its information.

Install dependencies with `pip install -r requirements.txt`.

Run `./fetch_plants.sh` to download the plant information needed for RAG.

Run `python3 plant_search_rag/plant_search_rag.py` and `python3 return_policy_rag/return_policy_rag.py` to generate RAG embeddings.