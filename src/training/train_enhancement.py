import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.dataset.loader import get_train_val_test_datasets
from src.models.enhancement_model import CustomUNet
from src.models.losses import CompositeLoss

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
    # Setup directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('docs', exist_ok=True)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 40
    batch_size = 4 #8 
    learning_rate = 1e-4
    
    # Load stable datasets split by source scan
    train_ds, val_ds, _ = get_train_val_test_datasets(
        raw_scans_dir="data/raw_scans",
        backgrounds_dir="data/backgrounds",
        target_size=(512, 512),
        epoch_length=800
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize model, loss and optimizer
    model = CustomUNet(in_channels=3, out_channels=3).to(device)
    
    # NEW: Load previous checkpoint if exists to save Colab GPU Quota
    checkpoint_path = 'checkpoints/enhancement_best.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[Checkpoint] Resuming training from pre-trained weights: {checkpoint_path}")
        learning_rate = 1e-5  # Use a lower learning rate for fine-tuning
    else:
        print("[Checkpoint] No pre-trained weights found. Training from scratch.")
        learning_rate = 1e-4
        
    criterion = CompositeLoss(alpha=0.4, beta=0.4, gamma=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    
    print("Starting training loop...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        
        print(f"Epoch [{epoch}/{epochs}] -> Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        # Save best model checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/enhancement_best.pth')
            print("==> New best model saved!")
            
    # Plot loss curves and save to docs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_loss_history, label='Train Loss')
    plt.plot(range(1, epochs + 1), val_loss_history, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Composite Loss')
    plt.title('Enhancement Network - Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig('docs/enhancement_training_loss.png')
    print("Training complete! Loss curves plotted to 'docs/enhancement_training_loss.png'")


if __name__ == "__main__":
    main()