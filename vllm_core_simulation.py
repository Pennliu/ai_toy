import numpy as np
import torch

# --- 1. PagedAttention（高效KV Cache管理）核心思想模拟 ---


class PagedKVCache:
    """
    模拟 vLLM 的 PagedAttention：将 KV Cache 分成小页，动态分配和复用，提升显存利用率。
    """

    def __init__(self, num_pages, page_size, hidden_dim):
        self.pages = np.zeros((num_pages, page_size, hidden_dim))
        self.page_used = [False] * num_pages

    def allocate(self, needed_tokens):
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
        for idx in page_indices:
            self.page_used[idx] = False


# --- 测试 PagedKVCache ---
if __name__ == "__main__":
    print("=== PagedAttention KV Cache 管理模拟 ===")
    cache = PagedKVCache(num_pages=8, page_size=4, hidden_dim=16)
    req1_pages = cache.allocate(6)  # 需要6 token，分配2页
    req2_pages = cache.allocate(8)  # 需要8 token，分配2页
    cache.free(req1_pages)  # 请求1结束，释放
    req3_pages = cache.allocate(4)  # 新请求复用空闲页
    print("当前已分配页：", [i for i, used in enumerate(cache.page_used) if used])

    # --- 2. 批量调度（高吞吐）核心思想模拟 ---
    print("\n=== 批量调度（高吞吐）模拟 ===")

    def batch_inference(model, prompts):
        max_len = max(len(p) for p in prompts)
        batch = [p + [0] * (max_len - len(p)) for p in prompts]
        batch_tensor = torch.tensor(batch, dtype=torch.float32)
        output = model(batch_tensor)
        return output

    prompts = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    model = torch.nn.Linear(4, 10)
    output = batch_inference(model, prompts)
    print("批量推理输出 shape:", output.shape)

    # --- 3. 流式输出（Streaming）核心思想模拟 ---
    print("\n=== 流式输出（Streaming）模拟 ===")

    def stream_generate(model, input_ids, max_tokens=5):
        output = input_ids[:]
        for _ in range(max_tokens):
            # 假设模型输出全1
            next_token = (
                model(torch.tensor([output], dtype=torch.float32))[0].argmax().item()
            )
            output.append(next_token)
            yield next_token

    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.ones(x.shape[1], 10)

    for token in stream_generate(DummyModel(), [1, 2, 3]):
        print("流式输出 token:", token)
