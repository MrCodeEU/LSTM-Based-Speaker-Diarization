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


def ge2e_loss(embeddings, labels, w, b, device, epsilon=1e-8):
    # Calculate centroids for each speaker
    unique_labels = torch.unique(labels)
    centroids = torch.stack([embeddings[labels == speaker_label].mean(dim=0) for speaker_label in unique_labels])

    # L2 normalize centroids and embeddings
    centroids = centroids / (centroids.norm(dim=1, keepdim=True) + epsilon)
    embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + epsilon)

    # Calculate similarity matrix (cosine similarity)
    similarity_matrix = torch.mm(embeddings, centroids.t()) * w + b

    # Create a mask for the diagonal elements (same speaker)
    mask = labels.unsqueeze(1) == unique_labels.unsqueeze(0)

    # Positive loss: -log(sigmoid(similarity of correct speaker))
    positive_similarity = similarity_matrix.masked_select(mask)
    positive_loss = -torch.log(torch.sigmoid(positive_similarity) + epsilon).mean()

    # Negative loss: -log(1 - sigmoid(similarity of most similar incorrect speaker))
    masked_similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
    most_similar_incorrect_speaker_similarities = masked_similarity_matrix.max(dim=1)[0]
    negative_loss = -torch.log(1 - torch.sigmoid(most_similar_incorrect_speaker_similarities) + epsilon).mean()

    # Combine the positive and negative losses
    loss = (positive_loss + negative_loss) / 2
    return loss

def ge2e_loss_softmax(embeddings, labels, w, b, device, epsilon=1e-6):
    # Calculate centroids for each speaker
    unique_labels = torch.unique(labels)
    centroids = torch.stack([embeddings[labels == speaker_label].mean(dim=0) for speaker_label in unique_labels])

    # L2 normalize centroids and embeddings
    centroids = centroids / (centroids.norm(dim=1, keepdim=True) + epsilon)
    embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + epsilon)

    # Calculate similarity matrix (cosine similarity)
    similarity_matrix = torch.mm(embeddings, centroids.t()) * w + b

    # Apply softmax to the similarity matrix
    similarity_matrix = torch.softmax(similarity_matrix, dim=1)

    # Create a mask for the diagonal elements (same speaker)
    mask = labels.unsqueeze(1) == unique_labels.unsqueeze(0)

    # Positive loss: -log(softmax similarity of correct speaker)
    positive_similarity = similarity_matrix.masked_select(mask)
    positive_loss = -torch.log(positive_similarity + epsilon).mean()

    # Negative loss: -log(1 - softmax similarity of most similar incorrect speaker)
    masked_similarity_matrix = similarity_matrix.masked_fill(mask, 0)
    most_similar_incorrect_speaker_similarities = masked_similarity_matrix.max(dim=1)[0]
    negative_loss = -torch.log(1 - most_similar_incorrect_speaker_similarities + epsilon).mean()

    # Combine the positive and negative losses
    loss = (positive_loss + negative_loss) / 2
    return loss

def ge2e_loss_contrast(embeddings, labels, w, b, device, epsilon=1e-6):
    # Calculate centroids for each speaker
    unique_labels = torch.unique(labels)
    centroids = torch.stack([embeddings[labels == speaker_label].mean(dim=0) for speaker_label in unique_labels])

    # L2 normalize centroids and embeddings
    centroids = centroids / (centroids.norm(dim=1, keepdim=True) + epsilon)
    embeddings = embeddings / (embeddings.norm(dim=1, keepdim=True) + epsilon)

    # Calculate similarity matrix (cosine similarity)
    similarity_matrix = torch.mm(embeddings, centroids.t()) * w + b

    # Create a mask for the diagonal elements (same speaker)
    mask = labels.unsqueeze(1) == unique_labels.unsqueeze(0)

    # Positive loss: 1 - sigmoid(similarity of correct speaker)
    positive_similarity = similarity_matrix.masked_select(mask)
    positive_loss = 1 - torch.sigmoid(positive_similarity)

    # Negative loss: max(sigmoid(similarity of incorrect speakers))
    negative_similarity = similarity_matrix.masked_fill(mask, float('-inf')).max(dim=1)[0]
    negative_loss = torch.sigmoid(negative_similarity)

    # Combine the positive and negative losses
    loss = torch.mean(positive_loss + negative_loss)
    return loss
