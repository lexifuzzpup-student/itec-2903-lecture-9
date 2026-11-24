#!/bin/bash

source .env

mkdir -p plant_search_rag
wget "${BASE_API_URL}/search/price" -O plant_search_rag/plants.json