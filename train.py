import datetime

import numpy as np
import torch
import torch.utils.data
import torchinfo
import tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from config import Hyperparameters
from dataset import SpeakerDataset
from model import LSTMModel, ge2e_loss
from utils import extract_and_label_features


def main():
    # ---------------------- 0. Set the Hyperparameters --------------------- #
    # Define the model parameters
    hp = Hyperparameters()
    input_size = hp.input_size  # Assuming log mel filterbank energies of dimension 40
    hidden_size = hp.hidden_size
    num_layers = hp.num_layers
    projection_size = hp.projection_size
    # Training loop
    num_epochs = hp.num_epochs
    batch_size = hp.batch_size
    learning_rate = hp.learning_rate
    # ---------------------- 1. Load and Segment Audio ---------------------- #
    # Set the path to your training data_icsi directory
    train_dir = hp.dataset + "/train"
    # Set the path to your segments directory
    segments_dir = hp.dataset + "/segments"

    labels, log_mel_spectrograms = extract_and_label_features(hp, segments_dir, train_dir)

    # ---------------------- 2. Create the Dataset and DataLoader ---------------------- #
    # Convert labels to numeric and then create the dataset and data_icsi loader
    label_to_id = {label: i for i, label in enumerate(set(labels))}
    numeric_labels = [label_to_id[label] for label in labels]
    print("Unique numeric labels: ", set(numeric_labels))

    train_dataset = SpeakerDataset(log_mel_spectrograms, numeric_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ---------------------- 3. Initialize the Model, Loss, and Optimizer ---------------------- #
    # Create the model
    model = LSTMModel(input_size, hidden_size, num_layers, projection_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    w = nn.Parameter(torch.tensor(10.0))  # Initialize w as a learnable parameter
    b = nn.Parameter(torch.tensor(-5.0))  # Initialize b as a learnable parameter
    w.to(device)
    b.to(device)
    # Define the optimizer
    optimizer = torch.optim.Adam(list(model.parameters()) + [w, b], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True,
                                                           threshold=0.01)
    model.to(device)
    # print the model information
    torchinfo.summary(model, input_size=(batch_size, 40))

    # Train and save the best model with early stopping
    train_model(b, device, model, num_epochs, optimizer, scheduler, train_loader, w)


def train_model(b, device, model, num_epochs, optimizer, scheduler, train_loader, w, patience=5, delta=0.01):
    train_losses = []
    best_loss = np.inf
    patience_counter = 0
    best_model_path = f'best_model_{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}.pth'

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Wrap the train_loader with tqdm to create a progress bar
        for segments, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            segments = segments.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = model(segments)
            loss = ge2e_loss(embeddings, labels, w, b, device)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip to a maximum norm of 1.0

            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Loss: {epoch_loss:.4f}")

        scheduler.step(epoch_loss)  # Update the learning rate based on the epoch loss

        # Check if the current loss is the best we've seen so far
        if epoch_loss < best_loss - delta:
            best_loss = epoch_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with loss {best_loss:.4f}")
        else:
            patience_counter += 1

        # Early stopping if the patience counter exceeds the patience threshold
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Visualize the training loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Epochs")
    plt.show()

    print(f"Training completed. Best model saved to {best_model_path} with loss {best_loss:.4f}")


if __name__ == "__main__":
    main()
