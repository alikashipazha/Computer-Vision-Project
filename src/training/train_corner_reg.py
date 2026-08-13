import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.dataset.loader import get_train_val_test_datasets
from src.models.corner_models import DirectRegressionNet
from src.training.train_enhancement import EarlyStopping

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs = batch['raw_photo'].to(device)
        targets = batch['corners'].to(device).view(inputs.size(0), -1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    for batch in dataloader:
        inputs = batch['raw_photo'].to(device)
        targets = batch['corners'].to(device).view(inputs.size(0), -1)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)


def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    epochs = 40
    batch_size = 8
    
    train_ds, val_ds, _ = get_train_val_test_datasets(
        raw_scans_dir="data/raw_scans",
        backgrounds_dir="data/backgrounds",
        target_size=(512, 512),
        epoch_length=800
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = DirectRegressionNet(in_channels=3).to(device)
    criterion = nn.L1Loss().to(device)
    
    best_val_loss = float('inf')
    start_epoch = 1
    learning_rate = 1e-4
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint_path = 'checkpoints/corner_reg_best.pth'
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"[Checkpoint] Resumed from epoch {checkpoint['epoch']} with saved loss: {best_val_loss:.5f}")
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    
    early_stopping = EarlyStopping(patience=5, min_delta=1e-5)
    
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    actual_epochs = 0
    
    print("Starting coordinate regression (Approach A) training loop...")
    for epoch in range(start_epoch, epochs + 1):
        actual_epochs = epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        print(f"Epoch [{epoch}/{epochs}] -> Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # torch.save(model.state_dict(), checkpoint_path)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }
            torch.save(checkpoint, checkpoint_path)
            print("==> New best regression model saved!")
            
        if early_stopping(val_loss):
            print(f"[EarlyStopping] Terminating training early at epoch {epoch}.")
            break
            
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, actual_epochs + 1), val_loss_history, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('L1 Coordinate Loss')
    plt.title('Approach A (Direct Regression) - Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/corner_regression_loss.png')
    print("Approach A training complete.")


if __name__ == "__main__":
    main()