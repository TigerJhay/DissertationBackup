from openai import OpenAI

client = OpenAI(api_key = "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")

completion = client.chat.completions.create(
    model="gpt-4o mini",
    messages=[
        {"role": "User", "content": "What is Dissertation?"}
    ]
)

print(completion.choices[0].message)