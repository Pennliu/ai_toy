import numpy as np

# 超参数
np.random.seed(42)
d_model = 4  # 词向量维度
n_heads = 2
seq_len = 3  # 序列长度
vocab_size = 5
learning_rate = 0.1
num_epochs = 200

# QKV权重初始化（d_model, d_model）
W_Q = np.random.randn(d_model, d_model) * 0.01
W_K = np.random.randn(d_model, d_model) * 0.01
W_V = np.random.randn(d_model, d_model) * 0.01

# 输出层权重（d_model, vocab_size）
W_out = np.random.randn(d_model, vocab_size) * 0.01
b_out = np.zeros(vocab_size)

# 简单的输入（假设已embedding）和目标输出（one-hot）
X = np.random.randn(seq_len, d_model)  # (seq_len, d_model)
y_true = np.zeros(vocab_size)
y_true[2] = 1  # 假设目标是词表中第2个词


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def cross_entropy(pred, target):
    return -np.sum(target * np.log(pred + 1e-8))


for epoch in range(num_epochs):
    # 前向传播
    Q = X @ W_Q  # (seq_len, d_model)
    K = X @ W_K
    V = X @ W_V
    # 注意力分数
    scores = Q @ K.T / np.sqrt(d_model)  # (seq_len, seq_len)
    attn = softmax(scores)
    context = attn @ V  # (seq_len, d_model)
    # 取最后一个字的context向量做分类
    logits = context[-1] @ W_out + b_out  # (vocab_size,)
    probs = softmax(logits)
    loss = cross_entropy(probs, y_true)

    # 反向传播（只对W_out做简单梯度，QKV权重用数值梯度，便于理解）
    grad_logits = probs - y_true  # (vocab_size,)
    grad_W_out = np.outer(context[-1], grad_logits)  # (d_model, vocab_size)
    grad_b_out = grad_logits

    # 数值梯度（QKV）
    eps = 1e-5
    grad_W_Q = np.zeros_like(W_Q)
    grad_W_K = np.zeros_like(W_K)
    grad_W_V = np.zeros_like(W_V)
    for i in range(d_model):
        for j in range(d_model):
            for W, grad_W in zip([W_Q, W_K, W_V], [grad_W_Q, grad_W_K, grad_W_V]):
                orig = W[i, j]
                W[i, j] = orig + eps
                Qp = X @ W_Q
                Kp = X @ W_K
                Vp = X @ W_V
                scores_p = Qp @ Kp.T / np.sqrt(d_model)
                attn_p = softmax(scores_p)
                context_p = attn_p @ Vp
                logits_p = context_p[-1] @ W_out + b_out
                probs_p = softmax(logits_p)
                loss_p = cross_entropy(probs_p, y_true)
                grad_W[i, j] = (loss_p - loss) / eps
                W[i, j] = orig

    # 权重更新
    W_out -= learning_rate * grad_W_out
    b_out -= learning_rate * grad_b_out
    W_Q -= learning_rate * grad_W_Q
    W_K -= learning_rate * grad_W_K
    W_V -= learning_rate * grad_W_V

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}: loss={loss:.4f}")

print("训练后QKV权重：")
print("W_Q:\n", W_Q)
print("W_K:\n", W_K)
print("W_V:\n", W_V)
