import datetime

import numpy as np
import torch
import torch.utils.data
import torchinfo
import tqdm
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from config import Hyperparameters
from dataset import SpeakerDataset, MultipleSpeakersBatchSampler
from model import LSTMModel, ge2e_loss, ge2e_loss_softmax
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
    eval_dir = hp.train_eval_dataset + "/eval"
    # Set the path to your segments directory
    segments_dir = hp.dataset + "/segments"

    labels, log_mel_spectrograms = extract_and_label_features(hp, segments_dir, train_dir, train=True)
    # DOES NOT WORK FOR ICSI
    eval_labels, eval_log_mel_spectrograms = extract_and_label_features(hp, segments_dir, eval_dir, train=False)

    # ---------------------- 2. Create the Dataset and DataLoader ---------------------- #
    # Convert labels to numeric and then create the dataset and data_icsi loader
    label_to_id = {label: i for i, label in enumerate(set(labels))}
    numeric_labels = [label_to_id[label] for label in labels]
    print("Unique numeric labels: ", set(numeric_labels))

    # Convert eval labels to numeric
    eval_label_to_id = {label: i for i, label in enumerate(set(eval_labels))}
    eval_numeric_labels = [eval_label_to_id[label] for label in eval_labels]
    print("Unique numeric labels: ", set(eval_numeric_labels))

    train_dataset = SpeakerDataset(log_mel_spectrograms, numeric_labels)
    eval_dataset = SpeakerDataset(eval_log_mel_spectrograms, eval_numeric_labels)  # Create eval dataset
    # Create the speaker_to_indices dictionary
    speaker_to_indices = {}
    for idx, label in enumerate(numeric_labels):
        if label not in speaker_to_indices:
            speaker_to_indices[label] = []
        speaker_to_indices[label].append(idx)

    # Create eval speaker_to_indices dictionary
    eval_speaker_to_indices = {}
    for idx, label in enumerate(eval_numeric_labels):
        if label not in eval_speaker_to_indices:
            eval_speaker_to_indices[label] = []
        eval_speaker_to_indices[label].append(idx)

    # Create the custom batch sampler
    speakers_per_batch = hp.num_speakers
    samples_per_speaker = hp.num_segments
    batch_sampler = MultipleSpeakersBatchSampler(speaker_to_indices, speakers_per_batch, samples_per_speaker)
    eval_batch_sampler = MultipleSpeakersBatchSampler(eval_speaker_to_indices, speakers_per_batch, samples_per_speaker)
    train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
    eval_loader = DataLoader(eval_dataset, batch_sampler=eval_batch_sampler)  # Eval dataloader

    # ---------------------- 3. Initialize the Model, Loss, and Optimizer ---------------------- #
    # Create the model
    model = LSTMModel(input_size, hidden_size, num_layers, projection_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    w = nn.Parameter(torch.tensor(50.0))  # Initialize w as a learnable parameter
    b = nn.Parameter(torch.tensor(-20.0))  # Initialize b as a learnable parameter
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
    train_model(b, device, model, num_epochs, optimizer, scheduler, train_loader, eval_loader, w)


def train_model(b, device, model, num_epochs, optimizer, scheduler, train_loader, eval_loader, w, patience=10, delta=0.005):
    train_losses, eval_losses = [], []  # Track both train and eval losses
    best_loss = np.inf
    patience_counter = 0
    best_model_path = f'best_model_{datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")}.pth'

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        model.train()
        # Wrap the train_loader with tqdm to create a progress bar
        for segments, labels in tqdm.tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch"):
            segments = segments.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            embeddings = model(segments)
            # loss = ge2e_loss(embeddings, labels, w, b, device)
            loss = ge2e_loss_softmax(embeddings, labels, w, b, device)
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip to a maximum norm of 1.0

            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        train_losses.append(epoch_loss)
        print(f"Loss: {epoch_loss:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}, w: {w.item():.4f}, b: "
              f"{b.item():.4f}, Patience: {patience_counter}/{patience}")

        model.eval()  # Set the model to evaluation mode
        eval_loss = 0.0
        with torch.no_grad():  # No need to calculate gradients during evaluation
            for segments, labels in tqdm.tqdm(eval_loader, desc="Evaluating", unit="batch"):
                segments = segments.to(device)
                labels = labels.to(device)
                embeddings = model(segments)
                loss = ge2e_loss_softmax(embeddings, labels, w, b, device)
                eval_loss += loss.item()
        eval_loss /= len(eval_loader)
        eval_losses.append(eval_loss)
        print(f"Eval Loss: {eval_loss:.4f}")

        scheduler.step(eval_loss)  # Update the learning rate based on the epoch loss

        # Check eval loss for best model and early stopping
        if eval_loss < best_loss - delta:
            best_loss = eval_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved (based on eval loss) {best_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Visualize both training and evaluation losses
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(eval_losses) + 1), eval_losses, label="Evaluation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss vs. Epochs")
    plt.legend()
    plt.show()

    # save loss in csv
    np.savetxt('train_losses.csv', train_losses, delimiter=',')
    np.savetxt('eval_losses.csv', eval_losses, delimiter=',')

    print(f"Training completed. Best model saved to {best_model_path} with loss {best_loss:.4f}")


if __name__ == "__main__":
    main()
