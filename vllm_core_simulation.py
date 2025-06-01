import numpy as np
import torch

# --- 1. PagedAttention (Efficient KV Cache Management) Core Simulation ---


class PagedKVCache:
    """
    Simulates vLLM's PagedAttention: splits the KV cache into small pages, dynamically allocates and reuses them to improve memory efficiency.

    Attributes:
        pages (np.ndarray): The KV cache pages, 3D array.
        page_used (list): Whether each page is occupied.
    """

    def __init__(self, num_pages, page_size, hidden_dim):
        """
        Initialize the PagedKVCache.

        Args:
            num_pages (int): Number of pages.
            page_size (int): Number of tokens per page.
            hidden_dim (int): Hidden dimension for each token.
        """
        self.pages = np.zeros((num_pages, page_size, hidden_dim))
        self.page_used = [False] * num_pages

    def allocate(self, needed_tokens):
        """
        Allocate the required number of KV cache pages.

        Args:
            needed_tokens (int): Number of tokens to allocate.

        Returns:
            list: List of allocated page indices.

        Raises:
            RuntimeError: If there are not enough free pages.
        """
        num_needed = (needed_tokens + self.pages.shape[1] - 1) // self.pages.shape[1]
        alloc = []
        for i, used in enumerate(self.page_used):
            if not used:
                alloc.append(i)
                self.page_used[i] = True
                if len(alloc) == num_needed:
                    break
        if len(alloc) < num_needed:
            raise RuntimeError("Out of KV cache pages!")
        return alloc

    def free(self, page_indices):
        """
        Free the specified KV cache pages.

        Args:
            page_indices (list): List of page indices to free.
        """
        for idx in page_indices:
            self.page_used[idx] = False


# --- Test PagedKVCache ---
if __name__ == "__main__":
    print("=== PagedAttention KV Cache Management Simulation ===")
    # Create a KV cache with 8 pages, each page has 4 tokens, hidden_dim is 16
    cache = PagedKVCache(num_pages=8, page_size=4, hidden_dim=16)
    req1_pages = cache.allocate(6)  # Allocate 2 pages for 6 tokens
    req2_pages = cache.allocate(8)  # Allocate 2 pages for 8 tokens
    cache.free(req1_pages)  # Free pages for request 1
    req3_pages = cache.allocate(4)  # Reuse freed pages for a new request
    print(
        "Currently allocated pages:",
        [i for i, used in enumerate(cache.page_used) if used],
    )

    # --- 2. Batch Scheduling (High Throughput) Core Simulation ---
    print("\n=== Batch Scheduling (High Throughput) Simulation ===")

    def batch_inference(model, prompts):
        """
        Perform batch inference by padding prompts to the same length and feeding them to the model.

        Args:
            model (torch.nn.Module): The model for inference.
            prompts (list of list of int): Multiple input sequences.

        Returns:
            torch.Tensor: Model output.
        """
        max_len = max(len(p) for p in prompts)
        batch = [
            p + [0] * (max_len - len(p)) for p in prompts
        ]  # Pad shorter prompts with zeros
        batch_tensor = torch.tensor(batch, dtype=torch.float32)
        output = model(batch_tensor)
        return output

    prompts = [
        [1, 2, 3],
        [4, 5],
        [6, 7, 8, 9],
    ]  # Three input sequences of different lengths
    model = torch.nn.Linear(4, 10)  # Input length 4, output dimension 10
    output = batch_inference(model, prompts)
    print("Batch inference output shape:", output.shape)

    # --- 3. Streaming Output Core Simulation ---
    print("\n=== Streaming Output Simulation ===")

    def stream_generate(model, input_ids, max_tokens=5):
        """
        Stream token generation, yielding one token at a time.

        Args:
            model (torch.nn.Module): The model for generation.
            input_ids (list of int): Initial input token sequence.
            max_tokens (int): Maximum number of tokens to generate.

        Yields:
            int: The generated token at each step.
        """
        output = input_ids[:]
        for _ in range(max_tokens):
            # Assume the model always outputs all-ones
            next_token = (
                model(torch.tensor([output], dtype=torch.float32))[0].argmax().item()
            )
            output.append(next_token)
            yield next_token

    class DummyModel(torch.nn.Module):
        """
        Dummy model that outputs an all-ones tensor at each step.
        """

        def forward(self, x):
            """
            Forward pass, returns an all-ones tensor.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                torch.Tensor: All-ones tensor.
            """
            return torch.ones(x.shape[1], 10)

    # Test streaming generation, print each generated token
    for token in stream_generate(DummyModel(), [1, 2, 3]):
        print("Streaming output token:", token)
