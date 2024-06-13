import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, projection_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.projection = nn.Linear(hidden_size, projection_size)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        d_vector = hidden[-1]  # Get the last hidden state
        d_vector = self.projection(d_vector)  # Apply linear projection
        # l2 normalization will be done outside of model
        # d_vector = d_vector / torch.norm(d_vector, dim=-1, keepdim=True)
        return d_vector


def ge2e_loss(embeddings, labels, w, b, device, epsilon=1e-6):
    # Calculate centroids for each speaker
    centroids = []
    for speaker_label in torch.unique(labels):
        speaker_embeddings = embeddings[labels == speaker_label]
        centroid = speaker_embeddings.mean(dim=0)
        centroids.append(centroid)
    centroids = torch.stack(centroids)

    # L2 normalize centroids and embeddings
    centroids = centroids / (centroids.norm(dim=1, keepdim=True) + epsilon)
    embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + epsilon)

    # Calculate similarity matrix (cosine similarity)
    similarity_matrix = torch.mm(embeddings, centroids.t()) * w + b

    # Create a mask for the diagonal elements (same speaker)
    unique_labels = torch.unique(labels)
    mask = labels.unsqueeze(1) == unique_labels.unsqueeze(0)

    # Positive loss: -log(sigmoid(similarity of correct speaker))
    positive_loss = -torch.log(torch.sigmoid(similarity_matrix.masked_select(mask)) + epsilon)

    # Negative loss: -log(1 - sigmoid(similarity of most similar incorrect speaker))
    masked_similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
    most_similar_incorrect_speaker_similarities = masked_similarity_matrix.max(dim=1)[0]
    negative_loss = -torch.log(1 - torch.sigmoid(most_similar_incorrect_speaker_similarities) + epsilon)

    # Average the losses over the batch
    loss = (positive_loss.mean() + negative_loss.mean()) / 2

    return loss
