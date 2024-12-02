import google.generativeai as genai

def getAI(searchvalue):
    genai.configure(api_key="AIzaSyDgRaOiicnXJSx_GNtfvuNxKLhCDCDpHhQ")
    searchvalue = input (searchvalue)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(searchvalue)
    #res_place = Element('')
    print(response.text)