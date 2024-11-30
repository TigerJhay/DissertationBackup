import openai


openai.api_key = "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"

completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Give me 3 ideas for apps I could build with openai apis "}])

print(completion.choices[0].message.content)