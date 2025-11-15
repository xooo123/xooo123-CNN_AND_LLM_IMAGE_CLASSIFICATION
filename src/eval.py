import os, torch, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from data import make_dataloaders
from model import SimpleCNN
import seaborn as sns

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = SimpleCNN(num_classes=len(ckpt.get('class_names', ['Normal','COVID'])))
    model.load_state_dict(ckpt['model_state'])
    model.to(device).eval()
    return model, ckpt.get('class_names', ['Normal','COVID'])

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, class_names = load_checkpoint(args.checkpoint, device)
    _, _, test_loader, _ = make_dataloaders(args.data_root, batch_size=args.batch_size, image_size=args.image_size)
    all_preds=[]; all_targets=[]
    with torch.no_grad():
        for x,y in test_loader:
            x=x.to(device)
            out = model(x)
            preds = out.argmax(1).cpu().numpy().tolist()
            all_preds.extend(preds)
            all_targets.extend(y.numpy().tolist())
    print('Accuracy:', accuracy_score(all_targets, all_preds))
    print('F1 (weighted):', f1_score(all_targets, all_preds, average='weighted'))
    print('Classification report:')
    print(classification_report(all_targets, all_preds, target_names=class_names))
    cm = confusion_matrix(all_targets, all_preds)
    os.makedirs(args.out_dir, exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(args.out_dir, 'confusion_matrix.png'))
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=224)
    args = parser.parse_args()
    evaluate(args)
