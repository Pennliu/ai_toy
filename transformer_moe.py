import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Traditional Transformer Feed Forward Network (FFN) ---


class TraditionalFeedForward(nn.Module):
    """
    Traditional Transformer Feed Forward Network (FFN).
    Each input token will pass through these two linear layers.
    """

    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)  # First linear layer
        self.gelu = nn.GELU()  # Activation function
        self.linear2 = nn.Linear(ff_dim, embed_dim)  # Second linear layer

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)

        # All input tokens will pass through linear layer 1
        intermediate = self.linear1(x)
        intermediate = self.gelu(intermediate)

        # All input tokens will pass through linear layer 2
        output = self.linear2(intermediate)

        return output


# --- 2. MoE Transformer's MoE Layer ---


class Expert(nn.Module):
    """
    A "expert" network, which is actually a small FFN.
    It has the same structure as TraditionalFeedForward but as a component of MoE.
    """

    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ff_dim, embed_dim)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


class MoELayer(nn.Module):
    """
    Mixture-of-Experts (MoE) layer.
    Contains multiple expert FFNs and a gate network to select experts.
    """

    def __init__(self, embed_dim, ff_dim, num_experts, top_k):
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot be greater than num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        # Create multiple expert instances
        self.experts = nn.ModuleList(
            [Expert(embed_dim, ff_dim) for _ in range(num_experts)]
        )
        # Gate network, used to predict which expert each token should route to
        self.gate = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)

        batch_size, seq_len, embed_dim = x.shape

        # Flatten input to allow gate network to process each token
        flat_x = x.view(-1, embed_dim)  # (batch_size * seq_len, embed_dim)

        # 1. Gate network calculates scores for each expert
        gate_logits = self.gate(flat_x)  # (batch_size * seq_len, num_experts)

        # 2. Apply softmax to scores to get routing probabilities
        # (batch_size * seq_len, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 3. Select Top-K experts
        # top_k will return the top top_k values and their indices
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)

        # To correctly weight, normalize top_k_probs to sum to 1
        # Avoid division by zero, add a small epsilon
        normalizing_constant = top_k_probs.sum(dim=-1, keepdim=True) + 1e-9
        # (batch_size * seq_len, top_k)
        top_k_weights = top_k_probs / normalizing_constant

        # 4. Initialize output tensor
        output = torch.zeros_like(flat_x)  # (batch_size * seq_len, embed_dim)

        # 5. Route input to selected experts and perform weighted sum
        # Iterate over each (sample, token) combination
        for i in range(batch_size * seq_len):
            input_token_i = flat_x[i]  # Embedded vector of current token

            # Selected expert indices and weights for current token
            selected_expert_indices = top_k_indices[i]
            selected_expert_weights = top_k_weights[i]

            # Iterate over selected top_k experts
            for k_rank in range(self.top_k):
                expert_idx = selected_expert_indices[
                    k_rank
                ].item()  # Actual expert index
                weight = selected_expert_weights[k_rank]  # Corresponding weight

                # Activate and calculate selected expert
                # Note: Here we operate on a single token, so we need unsqueeze(0) to increase batch dimension
                expert_output = self.experts[expert_idx](
                    input_token_i.unsqueeze(0)
                )  # (1, embed_dim)

                # Multiply expert output by corresponding weight and accumulate to total output
                output[i] += expert_output.squeeze(0) * weight

        # Restore output shape to original (batch_size, seq_len, embed_dim)
        return output.view(batch_size, seq_len, embed_dim)


# --- Demonstration and Comparison ---


# Model parameters
embed_dim = 768  # Embedding dimension
ff_dim = 3072  # FFN's intermediate dimension (traditionally 4 times embed_dim)
num_experts = 8  # Total number of experts in MoE
top_k = 2  # Number of experts each token activates in MoE

batch_size = 2  # Batch size
seq_len = 10  # Sequence length

# Simulate input data
dummy_input = torch.randn(batch_size, seq_len, embed_dim)  # Randomly generate input

print("--- Traditional Transformer FFN vs. MoE Layer Comparison ---")

# --- Traditional FFN Demonstration ---
print("\n=== Traditional Transformer FFN ===")
traditional_ffn = TraditionalFeedForward(embed_dim, ff_dim)
print(
    f"Total parameter count of Traditional FFN: {sum(p.numel() for p in traditional_ffn.parameters()):,} parameters"
)

output_traditional = traditional_ffn(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output_traditional.shape}")
print(
    "Features: Every parameter in FFN is activated and calculated during every forward propagation."
)

# --- MoE Layer Demonstration ---
print("\n=== MoE Layer ===")
moe_layer = MoELayer(embed_dim, ff_dim, num_experts, top_k)
print(
    f"Total parameter count of MoE Layer (including all experts and gate network): "
    f"{sum(p.numel() for p in moe_layer.parameters()):,} parameters"
)

output_moe = moe_layer(dummy_input)
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output_moe.shape}")
print(f"Features: MoE layer has {num_experts} experts.")
print(f"Each input token activates only {top_k} experts.")
print(
    f"This means that only about ({top_k}/{num_experts}) * 100% = ({
        top_k / num_experts:.2f}) * 100% of expert parameters are activated during each forward propagation step."
)
print(
    "Although MoE's total parameter count may be much greater than traditional FFN, its sparse activation characteristics make it have lower actual computation during inference."
)
