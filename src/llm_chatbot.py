!pip install -q transformers accelerate

from transformers import pipeline

chatbot = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-3B-Instruct",
    device_map="auto"
)

messages = [{"role": "system", "content": "You are a helpful assistant."}]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break

    messages.append({"role": "user", "content": user_input})
    output = chatbot(messages, max_new_tokens=400)
    reply = output[0]["generated_text"][-1]["content"]
    messages.append({"role": "assistant", "content": reply})

    print("Bot:", reply)

