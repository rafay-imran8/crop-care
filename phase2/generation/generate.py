import os
from openai import OpenAI
from generation.prompt import build_prompt

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

def generate_answer(context_blocks, question):
    prompt = build_prompt(context_blocks, question)

    response = client.chat.completions.create(
        model="mistralai/mistral-7b-instruct:free",  # 🔒 FREE MODEL (no quota)
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()
