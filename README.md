# 📌 Urdu Children's Story Generation AI App

## 📖 Project Overview

This project implements a fully functional **Urdu Story Generation AI App** using classical probabilistic language modeling techniques.

Unlike modern LLM-based systems, this solution is built using:

- Custom Byte Pair Encoding (BPE) tokenizer  
- Trigram probabilistic language model (MLE + Interpolation)  
- FastAPI microservice backend  
- Containerization using Docker  
- CI/CD via GitHub Actions  
- Web-based frontend  
- Deployment using Railway  

The system generates short Urdu children's stories based on a user-provided starting phrase.

---

## 🎯 Overall Objective

The project is divided into structured phases to bridge classical NLP techniques with modern software engineering practices.

The system performs the following:

- Scrapes and preprocesses real-world Urdu stories  
- Trains a custom BPE tokenizer (vocab size = 250)  
- Implements a Trigram Language Model with interpolation smoothing  
- Serves inference via containerized FastAPI microservice  
- Provides a ChatGPT-like web interface  
- Deploys backend using Railway  

---

User (Frontend)
↓

FastAPI Backend (Docker Container)
↓

Tokenizer + Trigram Model
↓

Generated Urdu Story


---

## ⚙️ Tech Stack

- Python 3.10+
- FastAPI
- Uvicorn
- Docker
- GitHub Actions
- Railway (Deployment)

---

## Install Dependencies
pip install -r requirements.txt

## Run FastAPI Server
py -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload

## Open in Browser

Then open:

http://localhost:8000

---

## 📦 Model Components

Custom BPE tokenizer (trained from scratch)

Trigram Language Model (MLE + Interpolation)

Special tokens: <EOS>, <EOP>, <EOT>

Temperature-controlled generation

JSON-serialized trained model for inference serving

---

## 👥 Contributors

Minahil Rizwan

Talha Akram

Abdullah Attique




## 🏗 System Architecture
