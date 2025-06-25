from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("my_api_key")
if api_key is None:
    print("API key not found!!")
    exit()

#connect to the API
client = OpenAI(
    api_key = api_key
)

# response from the bot
def health_query_response(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful medical assistant. Provide clear and friendly responses to health-related questions."
            " If the question involves serious medical advice, recommend consulting a healthcare professional."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()

if __name__ == "__main__":
    print("\n🩺 Hello Health Assistant AI Chatbot Feel free to ask for anything concering your health\n")
    while True:
        user_query = input("User: ")
        if user_query.lower() in ["exit", "quit", "bye", "q"]:
            print("\nAssistant: Take care! 👋")
            break

        health_bot_response = health_query_response(user_query)
        print(f"Assistant: {health_bot_response}\n")
