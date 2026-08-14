import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.dataset.loader import get_train_val_test_datasets
from src.models.enhancement_model import CustomUNet
from src.models.losses import CompositeLoss

class EarlyStopping:
    """
    Early stopping helper to terminate training when validation loss plateaus.
    """
    def __init__(self, patience: int = 5, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"[EarlyStopping] Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in dataloader:
        inputs = batch['rectified_input'].to(device)
        targets = batch['rectified_target'].to(device)
        
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
        inputs = batch['rectified_input'].to(device)
        targets = batch['rectified_target'].to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(dataloader.dataset)


def main():
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 40
    batch_size = 2  # Adjusted to 2 to prevent memory saturation on 6GB VRAM
    
    # Load stable datasets split by source scan
    train_ds, val_ds, _ = get_train_val_test_datasets(
        raw_scans_dir="data/raw_scans",
        backgrounds_dir="data/backgrounds",
        target_size=(512, 512),
        epoch_length=800
    )
    
    # For local Windows execution, num_workers must be 0 to prevent multiprocessing deadlocks
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = CustomUNet(in_channels=3, out_channels=3).to(device)
    criterion = CompositeLoss(alpha=0.4, beta=0.4, gamma=0.2).to(device)
    
    # Setup default state parameters
    best_val_loss = float('inf')
    start_epoch = 1
    learning_rate = 1e-4
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Load complete checkpoint if exists to resume training state fully
    checkpoint_path = 'checkpoints/enhancement_best.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['loss']
        start_epoch = checkpoint['epoch'] + 1
        print(f"[Checkpoint] Resumed from epoch {checkpoint['epoch']} with saved loss: {best_val_loss:.5f}")
        
        # Adjust learning rate dynamically for fine-tuning while preserving optimizer momentum
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-5
    
    # Instantiate early stopping mechanism
    early_stopping = EarlyStopping(patience=5, min_delta=1e-5)
    
    train_loss_history = []
    val_loss_history = []
    actual_epochs = 0
    
    print("Starting training loop...")
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
            print("==> New best model saved!")
            
        # Check early stopping condition
        if early_stopping(val_loss):
            print(f"[EarlyStopping] Terminating training early at epoch {epoch}.")
            break
            
    # Plot loss curves and save to docs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, actual_epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, actual_epochs + 1), val_loss_history, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Composite Loss')
    plt.title('Enhancement Network - Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/enhancement_training_loss.png')
    print("Training process finalized.")


if __name__ == "__main__":
    main()