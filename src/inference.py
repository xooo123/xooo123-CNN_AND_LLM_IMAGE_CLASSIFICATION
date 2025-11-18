import torch, os, json
from PIL import Image
from torchvision import transforms
from model import SimpleCNN
from llm_wrapper import explain_prediction

def load_image(path, image_size=224):
    img = Image.open(path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = SimpleCNN(num_classes=len(ckpt.get('class_names', ['Normal','COVID'])))
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model, ckpt.get('class_names', ['Normal','COVID'])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--llm', action='store_true', help='Call LLM for explanation (requires config)')
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, class_names = load_checkpoint(args.checkpoint, device)
    img = load_image(args.image, image_size=args.image_size).to(device)
    with torch.no_grad():
        out = model(img)
        pred = int(out.argmax(1).cpu().item())
        probs = torch.nn.functional.softmax(out, dim=1).cpu().numpy().tolist()[0]
    print(f"Predicted: {class_names[pred]} | Probabilities: {probs}")
    if args.llm:
        explanation = explain_prediction(args.image, class_names[pred], probs)
        print('\nLLM explanation:\n', explanation)
