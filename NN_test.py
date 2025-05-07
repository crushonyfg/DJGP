import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from JumpGP_test import load_dataset, compute_metrics

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=(8,)):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [N]

def parse_args():
    parser = argparse.ArgumentParser(description='Test MLP baseline with early stopping')
    parser.add_argument('--folder_name', type=str, required=True,
                        help='Folder containing dataset.pkl')
    parser.add_argument('--hidden', type=int, nargs='+', default=[64,64],
                        help='Hidden layer sizes, e.g. --hidden 64 32')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Fraction of training data to use as validation')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience on validation loss')
    return parser.parse_args()

def main():
    args = parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    dataset = load_dataset(args.folder_name)
    X_full = dataset["X_train"].to(device)   # [N_train, D]
    Y_full = dataset["Y_train"].to(device)   # [N_train]
    X_test  = dataset["X_test"].to(device)   # [N_test, D]
    Y_test  = dataset["Y_test"].to(device)   # [N_test]

    # Split train/validation
    N = X_full.size(0)
    val_size = int(args.val_ratio * N)
    train_size = N - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        torch.utils.data.TensorDataset(X_full, Y_full),
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    # Model, optimizer, loss
    D = X_full.shape[1]
    model = MLPRegressor(input_dim=D, hidden_dims=tuple(args.hidden)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # Training with early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        # Train epoch
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / train_size

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss_sum += criterion(pred, yb).item() * xb.size(0)
        val_loss = val_loss_sum / val_size

        print(f"Epoch {epoch}: Train MSE={train_loss:.4f}, Val MSE={val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    train_time = time.time() - start_time

    # Final evaluation on test set
    model.eval()
    with torch.no_grad():
        mu_pred = model(X_test)
    # For CRPS, use train-set MSE as proxy variance
    mse_train = criterion(model(X_full), Y_full).item()
    sigma = torch.sqrt(torch.tensor(mse_train, device=device))
    rmse, mean_crps = compute_metrics(mu_pred, sigma.expand_as(mu_pred), Y_test)

    print(f"Done in {train_time:.1f}s  Test RMSE={rmse:.4f}, mean-CRPS={mean_crps:.4f}")

if __name__ == "__main__":
    main()
