

import torch
import torch.nn as nn


# 每一步详细说明
# 数据准备：
# 把字符串拆成字符，建立字符到数字的映射。
# 构造输入序列和目标字符对。
# 模型定义：
# RNN/Transformer 都用嵌入层把字符编号转成向量。
# RNN 用循环神经网络处理序列，Transformer 用自注意力机制处理序列。
# 最后用全连接层输出每个字符的概率分布。
# 训练：
# 用交叉熵损失函数，优化模型参数，让模型学会根据前面的字符预测下一个字符。
# 生成：
# 给定起始字符序列，模型预测下一个字符。
# 把新字符加到序列末尾，继续预测下一个，直到生成足够长的文本。


# 1. 数据准备
text = "hello world"
chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

# 构造训练样本（输入长度为4，预测下一个字符）
seq_len = 4
data = []
for i in range(len(text) - seq_len):
    x_str = text[i:i + seq_len]
    y_str = text[i + seq_len]
    x = torch.tensor([stoi[ch] for ch in x_str])
    y = torch.tensor(stoi[y_str])
    data.append((x, y))

# 2. 定义模型


class CharTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embed(x) + self.pos_embed  # (batch, seq_len, embed_dim)
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)
        out = self.transformer(x)  # (seq_len, batch, embed_dim)
        out = out[-1, :, :]       # 取最后一个token
        logits = self.fc(out)     # (batch, vocab_size)
        return logits


# 3. 训练模型
model = CharTransformer(vocab_size, embed_dim=8, num_heads=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

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

# 4. 文本生成
# 给定前4个字符，生成后续字符
model.eval()
input_str = "hell"
input_idx = [stoi[ch] for ch in input_str]
input_tensor = torch.tensor(input_idx).unsqueeze(0)  # (1, seq_len)
generated = list(input_str)

for _ in range(8):
    logits = model(input_tensor)
    next_idx = torch.argmax(logits, dim=-1).item()
    next_char = itos[next_idx]
    generated.append(next_char)
    input_tensor = torch.tensor([stoi[ch]
                                for ch in generated[-seq_len:]]).unsqueeze(0)

print("Transformer生成结果:", "".join(generated))
