# PLAYDEAS: RAG Games Recommender (v0.1)
# Backend: fastapi(UNDER CONSTRUCTION)
# Frontend: Streamlit(done), Flutter (UNDER CONSTRUCTION)

## Table of Contents
- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Setup-streamlit](#setup-streamlit)
- [Setup-fastapinflutter](#setup-fastapinflutter)
- [Usage](#usage)

## Overview
This Streamlit application leverages Docker for deployment and integrates with various AI models and services. The application provides an interactive user interface for querying and interacting with data stored in a vector store.

## Folder Structure
.
├── fastapi
└── streamlit
    ├── app.py
    ├── data
    │   └── ollama
    │       ├── history
    │       └── models
    │           ├── blobs
    │           └── manifests
    │               └── registry.ollama.ai
    │                   └── library
    │                       ├── gemma2
    │                       │   └── latest
    │                       └── phi
    │                           └── latest
    ├── docker-compose.yml
    ├── Dockerfile
    ├── faiss_storage_1500_300
    │   ├── index.faiss
    │   └── index.pkl
    ├── logo
    │   ├── fastapi.png
    │   ├── flutter.png
    │   ├── langchain.jpg
    │   ├── logo.png
    │   ├── ollama.jpg
    │   └── streamlit.png
    ├── pages
    │   ├── homepage.py
    │   ├── querypage.py
    │   └── stackpage.py
    ├── README.md
    ├── requirements.txt
    └── template.txt


## Setup-streamlit
1. **Clone the repository**:
   ```bash
   git clone https://github.com/Tintanaa/rag_back
   cd streamlit

2. **Install Docker**: 
   Ensure docker is installed on your system. 
   For GPU leverage: 
   ### Windows 
   - Use Docker Desktop with WSL2: https://docs.docker.com/desktop/features/wsl/ https://docs.docker.com/desktop/features/gpu/
   ### Linux (Arch tested, NVIDIA)
   - https://medium.com/@srpillai/how-to-run-ollama-locally-on-gpu-with-docker-a1ebabe451e0 
   Also ensure that Nvidia-Container-Toolkit installed: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

3. **Pull necessary images**
   for LLM images: https://ollama.com/ 
   "pull some ones command": 
   docker exec -it name-of-container ollama run gemma2

4. **Build and Start the Application**:
   ```bash
   cd streamlit
   docker compose up --build

## Setup-fastapinflutter
### UNDER CONSTRUCTION