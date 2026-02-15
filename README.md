## 📌 Project Overview

This project implements a fully functional **Urdu Story Generation AI App** using classical probabilistic language modeling techniques.

Unlike modern LLM-based systems, this solution is built using:

- Custom Byte Pair Encoding (BPE) tokenizer  
- Trigram probabilistic language model (MLE + Interpolation)  
- FastAPI microservice backend  
- Containerization using Docker  
- CI/CD via GitHub Actions  
- Web-based frontend deployed on Vercel  

The system generates short Urdu children's stories based on a user-provided starting phrase.

---

## 🎯 Overall Objective

The project is divided into structured phases to bridge classical NLP techniques with modern software engineering practices.

The system performs the following:

- Scrapes and preprocesses real-world Urdu stories  
- Trains a custom BPE tokenizer  
- Implements a Trigram Language Model  
- Serves inference via containerized microservice  
- Provides a ChatGPT-like web interface  
- Deploys frontend to Vercel  

---

## 🏗 System Architecture

User (Frontend - Vercel)
↓
FastAPI Backend (Docker Container)
↓
Tokenizer + Trigram Model
↓
Generated Urdu Story


---

## 👥 Contributors

- Minahil Rizwan  
- Talha Akram  
- Abdullah Attique  
