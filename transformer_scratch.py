import numpy as np

# --- 0. 准备阶段：词汇表和“训练好的”权重 ---

# 词汇表：将字映射到数字ID，以及反向映射
vocab = {
    "<pad>": 0,  # 填充符，用于对齐序列长度
    "小": 1,
    "明": 2,
    "爱": 3,
    "吃": 4,
    "苹": 5,
    "果": 6,
    "子": 7,
    "笔": 8,
    "桌": 9,
    "<sos>": 10,  # 开始符 (在更完整的解码器中会用到)
    "<eos>": 11,  # 结束符
}
idx_to_word = {v: k for k, v in vocab.items()}

# 模型的维度参数
d_model = 64  # 词向量（嵌入）的维度
n_heads = 4  # 多头注意力的头数
d_k = d_model // n_heads  # 每个注意力头的维度

# --- 模拟“训练好的”权重 ---
# 1. 词嵌入矩阵 (Embedding Layer)
# 假设每个字都对应一个 d_model 维度的向量。
# 在实际模型中，这些向量是学习出来的，这里我们用随机值模拟。
# shape: (词汇表大小, d_model)
word_embeddings = np.random.randn(len(vocab), d_model) * 0.1

# 2. 注意力机制的权重 (Multi-Head Attention Weights)
# Q, K, V 的线性变换矩阵
# shape: (d_model, d_model)
W_Q = np.random.randn(d_model, d_model) * 0.01
W_K = np.random.randn(d_model, d_model) * 0.01
W_V = np.random.randn(d_model, d_model) * 0.01
W_O = np.random.randn(d_model, d_model) * 0.01  # 多头拼接后的输出权重

# 3. 最终预测层的权重 (Output Linear Layer)
# 将上下文表示映射到词汇表大小的维度
# shape: (d_model, vocab_size)
output_linear_W = np.random.randn(d_model, len(vocab)) * 0.01
output_linear_b = np.zeros(len(vocab))


# --- 辅助函数 (来自上一个例子) ---
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(
        d_k
    )  # 假设输入是 (batch, seq_len, dim)

    if mask is not None:
        # mask 是 (batch, seq_len_q, seq_len_k)，用 False 遮蔽
        scores = np.where(mask, scores, -1e9)

    attention_weights = softmax(scores, axis=-1)
    output = np.matmul(attention_weights, V)
    return output, attention_weights


# --- 推理过程开始 ---

print("--- 推理过程开始 ---")

# 1. 输入处理：转换为ID和词嵌入
input_sentence = "小明爱吃苹"
input_words = list(input_sentence)
input_ids = [vocab[word] for word in input_words]
# (1, 5) -> 假设批次大小为1，序列长度为5
input_sequence_ids = np.array([input_ids])  # 用于批处理

# 获取输入序列的词嵌入 (Lookup)
# input_embeddings: (batch_size, seq_len, d_model)
input_embeddings = word_embeddings[input_sequence_ids]

print(f"原始输入句子: {input_sentence}")
print(f"输入字ID: {input_ids}")
print(f"输入词嵌入形状: {input_embeddings.shape}")
print("\n--- 步骤 1: Multi-Head Attention (自注意力) ---")

# 在联想下一个字时，我们会将整个输入序列作为 Q, K, V 的来源
# Q, K, V 的输入都是 input_embeddings
Q_input = input_embeddings
K_input = input_embeddings
V_input = input_embeddings

batch_size, seq_len, _ = input_embeddings.shape

# 应用线性变换 (W_Q, W_K, W_V)
# (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
Q = np.matmul(Q_input, W_Q)
K = np.matmul(K_input, W_K)
V = np.matmul(V_input, W_V)

# 拆分成多个头 (n_heads)
# (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k)
# -> 转置为 (batch_size, n_heads, seq_len, d_k)
Q_heads = Q.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
K_heads = K.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)
V_heads = V.reshape(batch_size, seq_len, n_heads, d_k).transpose(0, 2, 1, 3)

# 创建自回归掩码 (Decoder Self-Attention Mask)
# 在预测下一个字时，当前字（Query）不能“看到”序列中它后面的字
# 这个 mask 是一个下三角矩阵，True 表示允许关注，False 表示遮蔽
# For '小明爱吃苹', when processing '苹', it can only see '小', '明', '爱', '吃', '苹' itself.
# When processing '吃', it can only see '小', '明', '爱', '吃'.
# shape: (seq_len, seq_len)
subsequent_mask = np.tril(np.ones((seq_len, seq_len))).astype(bool)  # 下三角矩阵
# 对于批次中的每个样本和每个头，掩码都是一样的
# (1, 1, seq_len, seq_len)
expanded_mask = subsequent_mask[np.newaxis, np.newaxis, :, :]


# 对每个头执行缩放点积注意力，并收集结果
attn_outputs_per_head = []
all_attention_weights = []  # 用于观察权重
for i in range(n_heads):
    # Q_i, K_i, V_i: (batch_size, seq_len, d_k)
    # 这里的 mask 应该是 (seq_len, seq_len) 应用到每个头
    output_i, weights_i = scaled_dot_product_attention(
        Q_heads[:, i, :, :],
        K_heads[:, i, :, :],
        V_heads[:, i, :, :],
        expanded_mask[:, 0, :, :],
    )
    attn_outputs_per_head.append(output_i)
    all_attention_weights.append(weights_i)  # 每个头的权重

# 拼接多头的输出
# (batch_size, seq_len, n_heads * d_k) = (batch_size, seq_len, d_model)
concatenated_output = np.concatenate(attn_outputs_per_head, axis=-1)

# 最终的线性变换 (W_O)
# (batch_size, seq_len, d_model)
attention_output = np.matmul(concatenated_output, W_O)

print(f"注意力机制输出形状 (代表每个字融合了上下文信息): {attention_output.shape}")

# 观察“苹”字的注意力权重 (简化演示，只看一个头的)
# 我们看最后一个字 '苹' (索引 4) 对前面所有字的关注程度
# weights_to_display = all_attention_weights[0][0, 4, :] # 第一个头，第一个样本， '苹' 的查询
# print(f"\n‘苹’字 (索引4) 对前面字的注意力权重 (某个头):")
# for j, word_id in enumerate(input_ids):
#     print(f"  对 '{idx_to_word[word_id]}': {weights_to_display[j]:.4f}")

print("\n--- 步骤 2: 获取最终预测的“上下文表示” ---")
# 因为我们想预测“苹”后面的字，所以我们只需要“苹”字经过注意力机制后的输出表示。
# 这个就是我们之前说的“信息球”或“语义指纹”。
# 它融合了“苹”自身和它前面所有字（特别是“吃”）的信息。
# 我们取序列中最后一个字的输出 (即“苹”字的输出)
final_context_representation = attention_output[:, -1, :]  # (batch_size, d_model)
print(
    f"‘苹’字经过注意力后得到的最终上下文表示形状: {final_context_representation.shape}"
)

print("\n--- 步骤 3: 词汇表打分 (最终的线性层) ---")
# 现在，这个“上下文表示”要被“翻译官”处理，计算词汇表中每个字的得分。
# 这是通过另一个线性变换完成的： (上下文表示 @ output_linear_W) + output_linear_b

# logits: (batch_size, vocab_size)
logits = np.matmul(final_context_representation, output_linear_W) + output_linear_b
print(f"词汇表每个字的原始分数 (Logits) 形状: {logits.shape}")

# 打印一些字的原始分数 (作为演示)
print("\n--- 词汇表部分原始分数 (Logits) ---")
print(f"‘果’字分数: {logits[0, vocab['果']]:.4f}")
print(f"‘子’字分数: {logits[0, vocab['子']]:.4f}")
print(f"‘笔’字分数: {logits[0, vocab['笔']]:.4f}")
print(f"‘吃’字分数: {logits[0, vocab['吃']]:.4f}")


print("\n--- 步骤 4: 转换为概率 (Softmax) ---")
# 将原始分数转换为概率分布，所有概率加起来是1。
# predicted_probabilities: (batch_size, vocab_size)
predicted_probabilities = softmax(logits, axis=-1)
print(f"词汇表每个字的预测概率形状: {predicted_probabilities.shape}")

# 打印一些字的预测概率 (作为演示)
print("\n--- 词汇表部分预测概率 ---")
print(f"‘果’字概率: {predicted_probabilities[0, vocab['果']]:.4f}")
print(f"‘子’字概率: {predicted_probabilities[0, vocab['子']]:.4f}")
print(f"‘笔’字概率: {predicted_probabilities[0, vocab['笔']]:.4f}")
print(f"‘吃’字概率: {predicted_probabilities[0, vocab['吃']]:.4f}")

print("\n--- 步骤 5: 最终预测 ---")
# 找出概率最高的那个字
predicted_word_id = np.argmax(predicted_probabilities, axis=-1)[0]
predicted_word = idx_to_word[predicted_word_id]

print(f"模型预测的下一个字ID: {predicted_word_id}")
print(f"模型预测的下一个字: '{predicted_word}'")

# 请注意：由于我们使用的是随机初始化的权重，模型在这里预测出“果”字的概率很低，
# 甚至可能预测出其他不相关的字。这恰恰说明了**模型训练的重要性**。
# 只有经过大量数据的训练，这些权重才能真正学习到语言的模式，
# 从而让“苹”字对应的上下文表示能够准确地指向“果”字。
