from openai import OpenAI

# Function to consume the OpenAI API
def ask_chatgpt(message):
    if message:
        # Load key from file
        with open('./tools/api-key.txt') as f:
            text = f.readlines()
        api_key = text[0]
        client = OpenAI(api_key = api_key)
        
        # Get response
        response = client.responses.create(
          model="gpt-4o-mini",
          input=message
        )
        
        return response.output_text