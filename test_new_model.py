from google import genai

client = genai.Client(api_key="AIzaSyD-Nw9rDFIa5mLgtjrPvlYHeW0Uh4i3jd8")
response = client.models.generate_content(
    model="gemini-2.0-flash", contents="How many people in the world?"  
)
print(response.text)