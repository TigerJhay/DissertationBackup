import openai
API_KEY = ("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")

openai.api_key = API_KEY

#chat_log = []

while True:
    user_msg = input()
    if user_msg.lower() == "quit":
        break
    else:
        #chat_log.append({"role":"user", "content":user_msg})
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[{"role":"user", "content":user_msg}]
        )
        message_response = response.choices[0].message.content.strip()
        print("The result is:", message_response.strip("\n").strip())