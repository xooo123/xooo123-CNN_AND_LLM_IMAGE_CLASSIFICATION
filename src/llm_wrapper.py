import os
def explain_prediction(image_path, predicted_label, probs):
    # Attempt OpenAI first (if available), else return a templated explanation.
    openai_key = os.environ.get('OPENAI_API_KEY', None)
    prompt = f"""You are an assistant that writes short, cautious medical-style explanations.
Image: {image_path}
Model prediction: {predicted_label}
Model probabilities: {probs}

Write a concise (3-6 sentence) explanation of what the model predicted, what signs the model might have used , and a short caution recommending clinical confirmation and PCR testing when appropriate."""
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            resp = openai.ChatCompletion.create(
                model='gpt-5o-mini', 
                messages=[{'role':'user','content':prompt}],
                max_tokens=200,
                temperature=0.2
            )
            return resp['choices'][0]['message']['content'].strip()
        except Exception as e:
            return f"(LLM call failed: {e})\n\nTemplate:\n" + _template(predicted_label, probs)
    else:
        return _template(predicted_label, probs)

def _template(predicted_label, probs):
    return (f"The model predicts '{predicted_label}' with probabilities {probs}. " 
            "This prediction is based on image patterns the CNN learned during training (opacities or consolidation). " 
            "This output is for educational/demonstration purposes only — clinical diagnosis requires PCR testing and expert radiologist review.") 
