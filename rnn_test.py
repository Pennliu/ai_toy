import torch
import torch.nn as nn

# Step-by-step explanation:
# Data preparation:
#   Split the string into characters and build a char-to-index mapping.
#   Construct input sequences and target character pairs.
# Model definition:
#   Both RNN/Transformer use an embedding layer to convert character indices to vectors.
#   RNN uses a recurrent neural network to process the sequence; Transformer uses self-attention.
#   A fully connected layer outputs the probability distribution for each character.
# Training:
#   Use cross-entropy loss to optimize model parameters so the model learns to predict the next character.
# Generation:
#   Given a starting character sequence, the model predicts the next character.
#   The new character is appended to the sequence, and prediction continues until enough text is generated.

# 1. Data preparation
text = "hello world"
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

# Construct training samples (input length 4, predict the next character)
seq_len = 4
data = []
for i in range(len(text) - seq_len):
    x_str = text[i : i + seq_len]
    y_str = text[i + seq_len]
    x = torch.tensor([stoi[ch] for ch in x_str])
    y = torch.tensor(stoi[y_str])
    data.append((x, y))


class CharRNN(nn.Module):
    """
    Character-level RNN for sequence prediction.

    Args:
        vocab_size (int): Number of unique characters.
        embed_dim (int): Embedding dimension.
        hidden_dim (int): Hidden state dimension.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """
        Forward pass for the RNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len).

        Returns:
            torch.Tensor: Logits for each character in the vocabulary.
        """
        x = self.embed(x)  # (batch, seq_len, embed_dim)
        out, _ = self.rnn(x)  # (batch, seq_len, hidden_dim)
        out = out[:, -1, :]  # Take the last time step
        logits = self.fc(out)  # (batch, vocab_size)
        return logits


# 3. Train the model
model = CharRNN(vocab_size, embed_dim=8, hidden_dim=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

# Simple training for 10 epochs
for epoch in range(10):
    total_loss = 0
    for x, y in data:
        x = x.unsqueeze(0)  # (1, seq_len)
        y = y.unsqueeze(0)  # (1,)
        logits = model(x)
        loss = loss_fn(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


# 4. Text generation
# Given the first 4 characters, generate subsequent characters
def generate_text(model, start_str, length=8):
    """
    Generate text using the trained RNN model.

    Args:
        model (nn.Module): Trained CharRNN model.
        start_str (str): Initial string to start generation.
        length (int): Number of characters to generate.

    Returns:
        str: Generated text.
    """
    model.eval()
    input_idx = [stoi[ch] for ch in start_str]
    input_tensor = torch.tensor(input_idx).unsqueeze(0)  # (1, seq_len)
    generated = list(start_str)
    for _ in range(length):
        logits = model(input_tensor)
        next_idx = torch.argmax(logits, dim=-1).item()
        next_char = itos[next_idx]
        generated.append(next_char)
        # Update input, sliding window
        input_tensor = torch.tensor(
            [stoi[ch] for ch in generated[-seq_len:]]
        ).unsqueeze(0)
    return "".join(generated)


print("RNN generated result:", generate_text(model, "hell", length=8))
