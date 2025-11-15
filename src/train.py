import argparse, os, random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from data import make_dataloaders
from model import SimpleCNN
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []
    preds = []
    targets = []
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.extend(out.argmax(1).detach().cpu().numpy().tolist())
        targets.extend(y.detach().cpu().numpy().tolist())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    return np.mean(losses), acc, f1

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            losses.append(loss.item())
            preds.extend(out.argmax(1).detach().cpu().numpy().tolist())
            targets.extend(y.detach().cpu().numpy().tolist())
    acc = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average='weighted')
    return np.mean(losses), acc, f1, preds, targets

def plot_curves(train_hist, val_hist, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, len(train_hist['loss'])+1))
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_hist['loss'], label='train_loss')
    plt.plot(epochs, val_hist['loss'], label='val_loss')
    plt.legend(); plt.title('Loss')
    plt.subplot(1,2,2)
    plt.plot(epochs, train_hist['acc'], label='train_acc')
    plt.plot(epochs, val_hist['acc'], label='val_acc')
    plt.legend(); plt.title('Accuracy')
    plt.savefig(os.path.join(out_dir, 'training_curves.png'))
    plt.close()

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, test_loader, class_names = make_dataloaders(args.data_root, batch_size=args.batch_size, image_size=args.image_size)
    model = SimpleCNN(num_classes=len(class_names), dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Cosine annealing with warm restarts for better optimization
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    best_val = 0.0
    train_hist = {'loss':[], 'acc':[], 'f1':[]}
    val_hist = {'loss':[], 'acc':[], 'f1':[]}
    os.makedirs(args.save_dir, exist_ok=True)

    patience = args.patience
    wait = 0
    for epoch in range(1, args.epochs+1):
        train_loss, train_acc, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = validate(model, val_loader, criterion, device)
        train_hist['loss'].append(train_loss); train_hist['acc'].append(train_acc); train_hist['f1'].append(train_f1)
        val_hist['loss'].append(val_loss); val_hist['acc'].append(val_acc); val_hist['f1'].append(val_f1)
        scheduler.step(val_loss)
        print(f"Epoch {epoch}: Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")
        # Early stopping on validation accuracy
        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model_state': model.state_dict(), 'class_names': class_names},os.path.join(args.save_dir, 'last.pth'))

            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print('Early stopping triggered.')
                break

    # Save training curves
    plot_curves(train_hist, val_hist, args.save_dir)
    print('Training complete. Best val acc:', best_val)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
