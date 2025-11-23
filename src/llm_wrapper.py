import os
import json
import base64
import requests
# ============================================================
# MAIN REASONING FUNCTION
# ============================================================
def explain_prediction(reasoning_input):
    """
    reasoning_input is a dict:
    {
        "image_path": ...,
        "predicted_label": ...,
        "probabilities": {...},
        "top1_prob": float,
        "top2_prob": float,
        "confidence_gap": float,
        "notes": ...
    }
    """

    predicted_label = reasoning_input["predicted_label"]
    probs = reasoning_input["probabilities"]
    image_path = reasoning_input["image_path"]

    # -----------------------------
    # REAL COLLABORATION PROMPT
    # -----------------------------
    prompt = f"""
        You are an AI medical reasoning assistant collaborating with a CNN chest X-ray classifier.

        Below is the CNN output JSON:
        {json.dumps(reasoning_input, indent=2)}

        Your tasks:
        1. Interpret the probability distribution.
        2. Evaluate uncertainty using the confidence gap.
        3. Identify possible misclassification risk.
        4. Provide chest X-ray based reasoning ONLY (no symptoms).
        5. Decide whether the CNN prediction is reliable.
        6. Produce a radiology-style interpretation using typical COVID/viral pneumonia/normal findings.

        STRICT RULES — FOLLOW EXACTLY:
        - NEVER say “I cannot see the image,” “no details are provided,” or “without more information.”
        - NEVER mention patient symptoms (no cough/fever/breathing).
        - ALWAYS mention at least TWO chest X-ray radiographic cues, even if typical:
            ground-glass opacities,
            bilateral peripheral infiltrates,
            patchy consolidation,
            hazy lower-lobe opacities,
            lobar consolidation,
            clear lung fields.
        - If the image features are unknown, say:
            “Radiographic cues typical for this diagnosis include: …”
        and then list concrete radiographic patterns — NOT patient symptoms.
        - Speak confidently and avoid soft language like “might”, “perhaps”, “possibly”.
        - If confidence_gap > 0.3 → call the model confident.
        - If confidence_gap ≤ 0.3 → call the model uncertain.

        Write the answer using EXACTLY these sections:
        - CNN Probability Interpretation
        - Uncertainty Evaluation
        - Medical Reasoning (must include radiographic cues)
        - Possible Misclassification
        - Final Judgment
        """


    # ============================================================
    # 1) TRY OPENAI (new API)
    # ============================================================
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.2
        )
        return response.choices[0].message["content"]

    except Exception as e:
        print(f"⚠ OpenAI failed: {e}")
        print("Falling back to Ollama...")

    # ============================================================
    # 2) TRY OLLAMA LLAVA
    # ============================================================
    try:
        explanation = _ollama_llava_explanation(image_path, prompt)
        if explanation:
            return explanation
    except Exception as e:
        print(f"⚠ Ollama failed: {e}")

    # ============================================================
    # 3) FINAL FALLBACK → TEMPLATE
    # ============================================================
    return _template(predicted_label, probs)


# ============================================================
# OLLAMA BACKEND (LLaVA)
# ============================================================
def _ollama_llava_explanation(image_path, prompt):
    """Call local Ollama LLaVA model using NDJSON streaming."""

    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [b64_image]
    }

    response = requests.post(
        "http://localhost:11434/api/generate",
        stream=True,
        json=payload,
        timeout=90
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


# ============================================================
# TEMPLATE FALLBACK
# ============================================================
def _template(predicted_label, probs):
    return (
        f"The model predicts '{predicted_label}' with probabilities {probs}. "
    "This prediction is based on learned radiographic patterns such as opacities or consolidation. "
    "This explanation is automatically generated for educational use only — clinical confirmation, "
    "PCR testing, and expert radiologist evaluation are always required."
    )