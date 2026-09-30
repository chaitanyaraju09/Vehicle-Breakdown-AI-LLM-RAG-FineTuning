!pip install -q chromadb pypdf transformers accelerate


import re, chromadb
from pypdf import PdfReader
from google.colab import files
from transformers import pipeline

# Upload the PDF and save it in the database
name = list(files.upload().keys())[0]
text = "\n".join(p.extract_text() for p in PdfReader(name).pages)
chunks = re.split(r"\n(?=\d+\.\s)", text)
db = chromadb.Client().create_collection("docs")
db.add(documents=chunks, ids=[str(i) for i in range(len(chunks))])

# Load the free AI model
bot = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct", device_map="auto")


while True:
    q = input("You: ")
    if q == "exit":
        break
    info = "\n".join(db.query(query_texts=[q], n_results=2)["documents"][0])
    msg = [{"role": "user", "content": f"Answer only from this info. If the answer is not there, say 'I don't know'.\n\nInfo:\n{info}\n\nQuestion: {q}"}]
    print("Bot:", bot(msg, max_new_tokens=200)[0]["generated_text"][-1]["content"], "\n")

