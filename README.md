# AI-Assisted Vehicle Breakdown System

This repository contains the project code, report, output evidence, team details, and presentation photo for the **Vehicle Breakdown System** project.

## Project overview

The project compares three approaches for an AI assistant that helps drivers during vehicle breakdowns:

1. **Simple LLM chatbot** — Qwen2.5-3B-Instruct
2. **RAG chatbot** — Qwen2.5-1.5B-Instruct + ChromaDB
3. **Fine-tuned LLM + retrieval** — Qwen2.5-0.5B-Instruct + LoRA

The report describes the hybrid approach in which retrieval provides domain information and fine-tuning shapes the response style.

## Repository structure

```text
vehicle-breakdown-ai-github/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── llm_chatbot.py
│   ├── rag_chatbot.py
│   └── fine_tuning.py
├── knowledge_base/
│   └── README.md
├── docs/
│   ├── Project_Report.docx
│   ├── OUTPUT-FILE.docx
│   ├── Team_Member_Details.xlsx
│   └── Presentation_Photo.docx
└── assets/
    ├── Presentation_Photo/
    ├── Project_Report/
    └── OUTPUT-FILE/
```

## Running the code

The supplied code is written for **Google Colab** and uses notebook-style `!pip install` commands and `google.colab.files` for PDF upload.

### 1. Simple LLM

Open `src/llm_chatbot.py` in Google Colab and run it. The model used is `Qwen/Qwen2.5-3B-Instruct`.

### 2. RAG chatbot

Add the Vehicle Breakdown Assistance Knowledge Base PDF to the Colab session and run `src/rag_chatbot.py`.

The RAG pipeline extracts PDF text with `pypdf`, stores chunks in ChromaDB, retrieves the most relevant chunks, and generates an answer with `Qwen/Qwen2.5-1.5B-Instruct`.

### 3. Fine-tuning + retrieval

Run `src/fine_tuning.py` in a GPU-enabled Colab session. It uses LoRA/PEFT and supervised fine-tuning with the Qwen2.5-0.5B-Instruct model, followed by retrieval from ChromaDB.

## Requirements

See `requirements.txt`. A GPU-enabled Google Colab runtime is recommended for the model and fine-tuning workloads.

## Important note

The repository contains the project prototype and documentation. The report states that the fine-tuning dataset contains only five examples and that the evaluation is qualitative; it should therefore be treated as a proof-of-concept rather than a production roadside-safety system.

## Team

- S SIDDU — VTU29420
- V CHAITANYA RAJU — VTU29435

## Project report

The full report contains the objectives, introduction, technology review, research gaps/questions, architecture, implementation, results, limitations, mobile-app proposal, impact analysis, cost analysis, commercialization plan, conclusion, and references.
