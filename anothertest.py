import google.generativeai as genai

genai.configure(api_key="AIzaSyDgRaOiicnXJSx_GNtfvuNxKLhCDCDpHhQ")
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Apple Smartwatch SE reviews")
print(response.text)