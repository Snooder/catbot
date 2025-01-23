import ollama

while True:
    user_input = input("You: ")
    response = ollama.chat(model="mistral", messages=[{"role": "user", "content": user_input}])
    print(f"Bot: {response['message']['content']}")