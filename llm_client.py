import argparse
import os
import base64
import requests
import sys
import json

# -----------------------------
# Local LLaVA (Ollama) functions
# -----------------------------
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_local_llm_reasoning(reasoning_input):
    """Send image + reasoning JSON to local LLaVA model via Ollama."""
    image_path = reasoning_input["image_path"]

    # Check Ollama server
    try:
        requests.get("http://localhost:11434", timeout=3)
    except requests.exceptions.RequestException:
        print("❌ Ollama server not running. Start it with: ollama serve")
        sys.exit(1)

    b64_image = encode_image(image_path)

    prompt = f"""
You are an AI medical reasoning assistant collaborating with a CNN classifier.

CNN output JSON:
{json.dumps(reasoning_input, indent=2)}

Tasks:
1. Interpret probability distribution.
2. Evaluate uncertainty using confidence_gap.
3. Mention at least two radiographic cues (ground-glass opacity, bilateral infiltrates, etc.).
4. Identify misclassification risk.
5. Give final diagnostic-style judgment.

Do NOT just rephrase the CNN label.
"""

    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [b64_image]
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        json=payload,
        stream=True,
        timeout=90
    )

    explanation = ""
    for line in response.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
            explanation += obj.get("response", "")
        except Exception:
            pass

    return explanation.strip()


# -----------------------------
# OpenAI functions (NEW API)
# -----------------------------
def get_openai_reasoning(reasoning_input):
    """Send reasoning JSON to OpenAI GPT for reasoning."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)

    prompt = f"""
You are an AI medical reasoning assistant collaborating with a CNN classifier.

CNN output JSON:
{json.dumps(reasoning_input, indent=2)}

Tasks:
1. Interpret probability distribution.
2. Evaluate uncertainty using confidence_gap.
3. Mention at least two radiographic cues (ground-glass opacity, bilateral infiltrates, etc.).
4. Identify misclassification risk.
5. Give final diagnostic-style judgment.

Do NOT just rephrase the CNN label.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get LLM reasoning for CNN predictions")
    parser.add_argument("--image", required=True, help="Path to chest X-ray image")
    parser.add_argument("--predicted_label", required=True, help="CNN predicted label")
    parser.add_argument("--probs", required=True, help="Comma-separated probabilities, e.g. 0.9,0.05,0.05")
    parser.add_argument("--class_names", required=True, help="Comma-separated class names, e.g. Covid,Normal,Viral Pneumonia")
    args = parser.parse_args()

    probs = [float(x) for x in args.probs.split(",")]
    class_names = [x.strip() for x in args.class_names.split(",")]

    top_probs = sorted(probs, reverse=True)
    top1, top2 = top_probs[0], top_probs[1]
    gap = top1 - top2

    reasoning_input = {
        "image_path": args.image,
        "predicted_label": args.predicted_label,
        "probabilities": {class_names[i]: probs[i] for i in range(len(class_names))},
        "top1_prob": top1,
        "top2_prob": top2,
        "confidence_gap": gap,
        "confidence_level": "high" if gap > 0.3 else "low",
        "notes": "SimpleCNN chest X-ray classifier"
    }

    # Try OpenAI first, fall back to local
    try:
        explanation = get_openai_reasoning(reasoning_input)
        backend_used = "openai"
    except Exception as e:
        print(f"⚠ OpenAI failed: {e}. Falling back to local LLaVA...")
        explanation = get_local_llm_reasoning(reasoning_input)
        backend_used = "local"

    print("\n====================================")
    print(f" LLM Reasoning ({backend_used})")
    print("====================================")
    print(explanation)

