import argparse
import os
import base64
import requests
import sys
import json  # IMPORTANT: needed for NDJSON parsing

# -----------------------------
# Local LLaVA (Ollama) functions
# -----------------------------
def encode_image(image_path):
    """Read an image file and encode to base64 for LLaVA."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_local_llm_explanation(image_path, prediction):
    """Send image + prediction to local LLaVA model via Ollama."""

    # Check if Ollama server is running
    try:
        requests.get("http://localhost:11434", timeout=3)
    except requests.exceptions.RequestException:
        print("❌ Ollama server is not running. Start it with:")
        print("   ollama serve")
        sys.exit(1)

    b64_image = encode_image(image_path)
    payload = {
        "model": "llava",
        "prompt": (
            f"The CNN predicted this chest X-ray as: {prediction}.\n"
            "Explain the reasoning behind this classification like a medical AI assistant, "
            "highlighting radiographic features, abnormalities, and uncertainty."
        ),
        "images": [b64_image]
    }

    # IMPORTANT: NDJSON streaming response
    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        stream=True
    )

    explanation = ""
    for line in response.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
            explanation += obj.get("response", "")
        except Exception:
            pass  # ignore malformed lines

    return explanation.strip()


# -----------------------------
# OpenAI functions
# -----------------------------
def get_openai_explanation(image_path, prediction):
    """Send image + prediction to OpenAI GPT for explanation."""
    try:
        import openai
    except ImportError:
        print("❌ Install OpenAI package:  pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY") #add your openai key here

    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    openai.api_key = api_key

    prompt = (
        f"CNN predicted this chest X-ray as: {prediction}.\n"
        "Explain the reasoning behind this prediction like a medical AI assistant."
    )

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2,
        )
        return response.choices[0].message["content"]

    except Exception as e:
        print(f"❌ OpenAI API error: {e}")
        sys.exit(1)


# -----------------------------
# Main script
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get LLM explanations for CNN predictions")
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--prediction", required=True, help="CNN predicted label")
    parser.add_argument(
        "--backend",
        choices=["local", "openai"],
        default="local",
        help="LLM backend: local (LLaVA) or openai"
    )
    args = parser.parse_args()

    if args.backend == "local":
        explanation = get_local_llm_explanation(args.image, args.prediction)
    else:
        explanation = get_openai_explanation(args.image, args.prediction)

    print("\n====================================")
    print(f" LLM Explanation ({args.backend})")
    print("====================================")
    print(explanation)
