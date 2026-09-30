!pip install -q transformers peft trl datasets accelerate chromadb pypdf

import re, chromadb
from pypdf import PdfReader
from google.colab import files

name = list(files.upload().keys())[0]
text = "\n".join(p.extract_text() for p in PdfReader(name).pages)
sections = [p.strip() for p in re.split(r"\n(?=\d+\.\s)", text) if re.match(r"\d+\.", p.strip())]

db = chromadb.Client().get_or_create_collection("docs")
db.add(documents=sections, ids=[str(i) for i in range(len(sections))])
print(db.count())   # should print 5

!pip install -q transformers==4.46.3 trl==0.11.4 peft==0.13.2 datasets accelerate

from datasets import Dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Fine-tuning examples
data = [
    {
        "text": "Question: My tyre is punctured.\nAnswer: Stop safely, turn on hazard lights, and contact roadside assistance."
    },
    {
        "text": "Question: My battery is dead.\nAnswer: Contact a mechanic for a battery check or jump-start."
    },
    {
        "text": "Question: My car is overheating.\nAnswer: Stop the vehicle, switch off the engine, let it cool, and contact a mechanic if needed."
    },
    {
        "text": "Question: My vehicle is not starting.\nAnswer: Check the fuel, battery, and warning lights. Contact a mechanic if it still does not start."
    },
    {
        "text": "Question: What information should I give for roadside assistance?\nAnswer: Give your location, vehicle model, problem, and contact number."
    }
]

dataset = Dataset.from_list(data)

lora = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    target_modules="all-linear"
)

trainer = SFTTrainer(
    model=model_name,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    peft_config=lora,
    args=SFTConfig(
        output_dir="fine_tuned_model",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=True
    )
)

trainer.train()

print("Fine-tuning completed!")

model = trainer.model
model.eval()

print("Fine-tuned model loaded!")


def ask(question):

    result = db.query(
        query_texts=[question],
        n_results=1
    )

    info = result["documents"][0][0]

    prompt = f"""
Use the information below to answer the question.

Information:
{info}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    print("Answer:", answer)


question = input("Ask: ")
ask(question)