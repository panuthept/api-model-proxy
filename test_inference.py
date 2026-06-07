from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="aisingapore/Gemma-SEA-LION-v4-27B-IT",
    messages=[{"role": "user", "content": "Hello, world!"}]
)
print(response.choices[0].message.content)