# llm_client.py
import os
import json
import requests
import openai  # pip install openai

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_SERVER_URL = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8000/predict")
MODEL_SERVER_KEY = os.environ.get("MODEL_SERVER_API_KEY", "dev-key")

openai.api_key = OPENAI_API_KEY

# Define the function schema you register with OpenAI
FUNCTIONS = [
    {
        "name": "call_covid_model",
        "description": "Analyze a chest X-ray image and return COVID-19 prediction",
        "parameters": {
            "type": "object",
            "properties": {
                "image_url": {"type": "string", "description": "URL to the chest X-ray image"}
            },
            "required": ["image_url"]
        },
    }
]

def call_openai_with_user_message(user_prompt: str):
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",  # replace by available model in your account
        messages=[{"role": "user", "content": user_prompt}],
        functions=FUNCTIONS,
        function_call="auto",
        temperature=0.2,
        max_tokens=400,
    )
    return resp

def handle_function_call(function_call):
    # parse args
    name = function_call["name"]
    args = json.loads(function_call["arguments"])
    if name == "call_covid_model":
        image_url = args.get("image_url")
        if not image_url:
            return {"error": "missing image_url"}

        # call your model server (it will download image if you pass image_url,
        # but we prefer to download and upload to avoid server-side fetching policies)
        try:
            img_bytes = requests.get(image_url, timeout=10).content
        except Exception as e:
            return {"error": f"failed to download image: {e}"}

        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        headers = {"x-api-key": MODEL_SERVER_KEY}
        r = requests.post(MODEL_SERVER_URL + "?use_llm=false", files=files, headers=headers, timeout=30)
        return r.json()
    else:
        return {"error": "unknown function"}

if __name__ == "__main__":
    # Example: ask LLM to analyze an image URL.
    user_question = ("Please analyze the chest X-ray at this URL and determine whether "
                     "the CNN would predict COVID-19. If needed call the function.")
    # append the actual URL in the prompt (or let the LLM ask for it)
    user_question += "\nImage URL: https://example.com/path/to/xray.jpg"

    openai_resp = call_openai_with_user_message(user_question)
    choice = openai_resp["choices"][0]
    if choice.get("finish_reason") == "function_call" or choice.get("message", {}).get("function_call"):
        func = choice["message"]["function_call"]
        result = handle_function_call(func)
        # Now send result back to OpenAI to let it create a final message (optional)
        followup = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": user_question},
                {"role": "assistant", "content": None, "function_call": func}, # pass function call
                {"role": "function", "name": func["name"], "content": json.dumps(result)}
            ],
            temperature=0.2,
            max_tokens=300
        )
        print("LLM final response:\n", followup["choices"][0]["message"]["content"])
    else:
        print("LLM responded without function call:", choice)
