import numpy as np

# 超参数
np.random.seed(42)
d_model = 8  # 词向量维度
n_heads = 2  # 多头数量
d_k = d_model // n_heads
seq_len = 4  # 序列长度

# 输入（假设已embedding）
X = np.random.randn(seq_len, d_model)  # (seq_len, d_model)

# 多头QKV权重（每个头独立一套参数）
W_Q = np.random.randn(n_heads, d_model, d_k) * 0.1  # (n_heads, d_model, d_k)
W_K = np.random.randn(n_heads, d_model, d_k) * 0.1
W_V = np.random.randn(n_heads, d_model, d_k) * 0.1

# 输出拼接后的线性变换
W_O = np.random.randn(n_heads * d_k, d_model) * 0.1  # (n_heads*d_k, d_model)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


# 1. 计算每个头的Q, K, V
Q = np.zeros((n_heads, seq_len, d_k))
K = np.zeros((n_heads, seq_len, d_k))
V = np.zeros((n_heads, seq_len, d_k))
for h in range(n_heads):
    Q[h] = X @ W_Q[h]  # (seq_len, d_k)
    K[h] = X @ W_K[h]
    V[h] = X @ W_V[h]

# 2. 每个头独立计算注意力输出
head_outputs = []
for h in range(n_heads):
    # (seq_len, d_k) @ (d_k, seq_len) -> (seq_len, seq_len)
    scores = Q[h] @ K[h].T / np.sqrt(d_k)
    attn = softmax(scores, axis=-1)
    out = attn @ V[h]  # (seq_len, d_k)
    head_outputs.append(out)

# 3. 拼接所有头的输出
concat = np.concatenate(head_outputs, axis=-1)  # (seq_len, n_heads*d_k)

# 4. 通过输出线性层
output = concat @ W_O  # (seq_len, d_model)

print("输入X shape:", X.shape)
print("每个头Q shape:", Q.shape)
print("拼接后 shape:", concat.shape)
print("最终输出 shape:", output.shape)
print("最终输出:\n", output)
