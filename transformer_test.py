import torch  # Import PyTorch main library
import torch.nn as nn  # Import neural network module

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
text = "hello world"  # The text to model
chars = sorted(list(set(text)))  # Get all unique characters in the text and sort them
stoi = {ch: i for i, ch in enumerate(chars)}  # Char-to-index mapping
itos = {i: ch for ch, i in stoi.items()}  # Index-to-char mapping
vocab_size = len(chars)  # Vocabulary size

# Construct training samples (input length 4, predict the next character)
seq_len = 4  # Input sequence length
data = []  # Store training samples
for i in range(len(text) - seq_len):  # Iterate over all possible substrings
    x_str = text[i:i + seq_len]  # Input substring
    y_str = text[i + seq_len]  # Target character
    x = torch.tensor([stoi[ch] for ch in x_str])  # Input as index tensor
    y = torch.tensor(stoi[y_str])  # Target as index
    data.append((x, y))  # Add to training data


# 2. Model definition
class CharTransformer(nn.Module):
    """
    Character-level Transformer model for sequence prediction.

    Args:
        vocab_size (int): Number of unique characters.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()  # Call parent constructor
        self.embed = nn.Embedding(vocab_size, embed_dim)  # Embedding layer
        self.pos_embed = nn.Parameter(
            torch.zeros(1, seq_len, embed_dim)
        )  # Positional embedding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads
        )  # Single-layer Transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1
        )  # Encoder stack
        self.fc = nn.Linear(embed_dim, vocab_size)  # Output layer

    def forward(self, x):
        """
        Forward pass for the Transformer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len).

        Returns:
            torch.Tensor: Logits for each character in the vocabulary.
        """
        x = (
            self.embed(x) + self.pos_embed
        )  # (batch, seq_len, embed_dim) with positional embedding
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim) for Transformer input
        out = self.transformer(x)  # (seq_len, batch, embed_dim) after Transformer
        out = out[-1, :, :]  # Take the output of the last token
        logits = self.fc(
            out
        )  # (batch, vocab_size) output probabilities for each character
        return logits  # Return prediction


# 3. Train the model
model = CharTransformer(vocab_size, embed_dim=8, num_heads=2)  # Instantiate model
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer
loss_fn = nn.CrossEntropyLoss()  # Cross-entropy loss

for epoch in range(10):  # Train for 10 epochs
    total_loss = 0  # Accumulate loss
    for x, y in data:  # Iterate over each training sample
        x = x.unsqueeze(0)  # (1, seq_len), add batch dimension
        y = y.unsqueeze(0)  # (1,)
        logits = model(x)  # Forward pass
        loss = loss_fn(logits, y)  # Compute loss
        optimizer.zero_grad()  # Zero gradients
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        total_loss += loss.item()  # Accumulate loss
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")  # Print loss for each epoch

# 4. Text generation
# Given the first 4 characters, generate subsequent characters
model.eval()  # Switch to evaluation mode
input_str = "hell"  # Initial input
input_idx = [stoi[ch] for ch in input_str]  # Convert to indices
input_tensor = torch.tensor(input_idx).unsqueeze(0)  # (1, seq_len)
generated = list(input_str)  # Generated character sequence

for _ in range(8):  # Generate 8 characters
    logits = model(input_tensor)  # Forward pass
    next_idx = torch.argmax(
        logits, dim=-1
    ).item()  # Get the index of the most probable next character
    next_char = itos[next_idx]  # Convert to character
    generated.append(next_char)  # Add to generated sequence
    input_tensor = torch.tensor([stoi[ch] for ch in generated[-seq_len:]]).unsqueeze(
        0
    )  # Update input

print(
    "Transformer generated result:", "".join(generated)
)  # Print the final generated text
