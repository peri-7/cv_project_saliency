import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from tqdm import tqdm

from src.dataset import LoraDataset
from src.models import MaeViT
from src.decoder import Decoder
from src.losses import Composite_Loss
from src.training_online import train_one_epoch_online, evaluate_model_online, test_model_online

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("---  Phase 1+2: Model Inference + Decoder Training (MAE ViT-B) ---")

# Same transforms as all other smoke tests: input normalization is held fixed
# across backbones for benchmark fairness.
image_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Ground truth standardization
map_transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])


train_dataset = LoraDataset(
    image_dir="mini_data/images/train",
    maps_dir="mini_data/maps/train",
    fixations_dir="mini_data/fixations/train",
    image_transform=image_transform,
    map_transform=map_transform
)
# Use a small batch size for testing (ViT-B is heavy, especially on CPU)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

val_dataset = LoraDataset(
    image_dir="mini_data/images/val",
    maps_dir="mini_data/maps/val",
    fixations_dir="mini_data/fixations/val",
    image_transform=image_transform,
    map_transform=map_transform
)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

# Swap ResNet/SamViT -> MaeViT; everything downstream is backbone-agnostic.
extractor = MaeViT().to(device)

# Architecture Initialization. hidden_dim is set to 256.
decoder = Decoder(in_channels_list=extractor.out_channels, hidden_dim=256).to(device)
criterion = Composite_Loss().to(device)
optimizer = optim.Adam(decoder.parameters(), lr=1e-4)
epochs = 4
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=epochs)

# Training Loop with Early Stopping
patience = 0
val_min = float('inf')

print('start training')
for epoch in range(epochs):

    current_lr = scheduler.get_last_lr()[0]

    train_loss, kld, cc = train_one_epoch_online(extractor, decoder, train_loader, optimizer, criterion, device)
    val_loss = evaluate_model_online(extractor, decoder, val_loader, criterion, device)

    print(f"Epoch {epoch+1} | LR: {current_lr:.3e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    scheduler.step()

    # Track the best validation loss
    if val_loss < val_min:
        patience = 0
        val_min = val_loss
    else:
        patience += 1

    if patience > 3:
        print(f"Early Stopping triggered on Epoch {epoch}. Restoring best weights.")
        break

print("-" * 50)
print("Training complete.")


print("--- Phase 3: Final Evaluation ---")

# Run the strict metric calculation on the validation proxy set
avg_loss, avg_kld, avg_cc, avg_sim, avg_nss, avg_auc, avg_ig = test_model_online(extractor, decoder, val_loader, criterion, device)

print(f"Final Model Benchmark (MAE ViT-B Backbone):")
print(f"Composite Loss: {avg_loss:.4f}")
print(f"KLD: {avg_kld:.4f}")
print(f"CC:  {avg_cc:.4f}")
print(f"SIM: {avg_sim:.4f}")
print(f"NSS: {avg_nss:.4f}")
print(f"AUC: {avg_auc:.4f}")
print(f"IG:  {avg_ig:.4f} bits")
