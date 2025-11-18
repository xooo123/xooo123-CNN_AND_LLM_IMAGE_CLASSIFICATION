import os
import json
import base64
import requests

def explain_prediction(image_path, predicted_label, probs):
    """
    Try OpenAI → if that fails, try Ollama LLaVA → if that fails, use template.
    """

    prompt = f"""You are an assistant that writes short, cautious medical-style explanations.
Image: {image_path}
Model prediction: {predicted_label}
Model probabilities: {probs}

Write a concise (3–6 sentence) explanation describing:
- why this label was likely predicted,
- what radiographic signs might have influenced the model,
- uncertainty,
- and a reminder that real diagnosis requires PCR + radiologist review.
"""

    # ----------------------------------------
    # 1) TRY OPENAI FIRST
    # ----------------------------------------
    openai_key = ("OPENAI_API_KEY")

    if openai_key:  # Only attempt if key actually set
        try:
            import openai
            openai.api_key = openai_key

            resp = openai.ChatCompletion.create(
                model="gpt-5o-mini",
                messages=[{"role":"user","content": prompt}],
                max_tokens=250,
                temperature=0.2
            )

            return resp["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"⚠ OpenAI failed: {e}")
            print("Falling back to Ollama.")

    # ----------------------------------------
    # 2) FALLBACK → TRY OLLAMA LLAVA
    # ----------------------------------------
    try:
        explanation = _ollama_llava_explanation(image_path, predicted_label)
        if explanation:
            return explanation
    except Exception as e:
        print(f"⚠ Ollama failed: {e}")

    # ----------------------------------------
    # 3) FINAL FALLBACK → TEMPLATE TEXT
    # ----------------------------------------
    return _template(predicted_label, probs)


# ------------------------------------------------
# OLLAMA LLAVA BACKEND
# ------------------------------------------------
def _ollama_llava_explanation(image_path, predicted_label):
    """Call local Ollama LLaVA model using NDJSON streaming."""
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "llava",
        "prompt": f"The CNN predicted: {predicted_label}. Explain the reasoning.",
        "images": [b64_image]
    }

    response = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=30
    )

    explanation = ""
    for line in response.iter_lines():
        if not line:
            continue
        try:
            obj = json.loads(line.decode("utf-8"))
            explanation += obj.get("response", "")
        except:
            pass

    return explanation.strip()


# ------------------------------------------------
# TEMPLATE (ONLY when all LLMs fail)
# ------------------------------------------------
def _template(predicted_label, probs):
    return (
        f"The model predicts '{predicted_label}' with probabilities {probs}. "
        "This prediction is based on learned radiographic patterns "
        "such as opacities or consolidation. "
        "This explanation is automatically generated for educational use only — "
        "clinical confirmation, PCR testing, and expert radiologist evaluation "
        "are always required."
    )


