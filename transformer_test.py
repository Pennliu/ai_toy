import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块


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
text = "hello world"  # 要建模的文本
chars = sorted(list(set(text)))  # 获取文本中所有唯一字符，并排序
stoi = {ch: i for i, ch in enumerate(chars)}  # 字符到索引的映射
itos = {i: ch for ch, i in stoi.items()}  # 索引到字符的映射
vocab_size = len(chars)  # 字符表大小

# 构造训练样本（输入长度为4，预测下一个字符）
seq_len = 4  # 输入序列长度
data = []  # 存放训练样本
for i in range(len(text) - seq_len):  # 遍历所有可用的子串
    x_str = text[i:i + seq_len]  # 输入子串
    y_str = text[i + seq_len]  # 目标字符
    x = torch.tensor([stoi[ch] for ch in x_str])  # 输入转为索引张量
    y = torch.tensor(stoi[y_str])  # 目标转为索引
    data.append((x, y))  # 加入训练数据


# 2. 定义模型
class CharTransformer(nn.Module):  # 定义字符级 Transformer 模型
    def __init__(self, vocab_size, embed_dim, num_heads):
        super().__init__()  # 调用父类构造函数
        self.embed = nn.Embedding(vocab_size, embed_dim)  # 嵌入层
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))  # 位置嵌入
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads)  # 单层 Transformer 编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=1)  # 编码器堆叠
        self.fc = nn.Linear(embed_dim, vocab_size)  # 输出层

    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embed(x) + self.pos_embed  # (batch, seq_len, embed_dim)，加上位置嵌入
        x = x.transpose(0, 1)  # (seq_len, batch, embed_dim)，适配 Transformer 输入
        out = self.transformer(x)  # (seq_len, batch, embed_dim)，经过 Transformer
        out = out[-1, :, :]  # 取最后一个 token 的输出
        logits = self.fc(out)  # (batch, vocab_size)，输出每个字符的概率
        return logits  # 返回预测结果


# 3. 训练模型
model = CharTransformer(vocab_size, embed_dim=8, num_heads=2)  # 实例化模型
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam 优化器
loss_fn = nn.CrossEntropyLoss()  # 交叉熵损失

for epoch in range(10):  # 训练 10 个 epoch
    total_loss = 0  # 累计损失
    for x, y in data:  # 遍历每个训练样本
        x = x.unsqueeze(0)  # (1, seq_len)，增加 batch 维
        y = y.unsqueeze(0)  # (1,)
        logits = model(x)  # 前向传播
        loss = loss_fn(logits, y)  # 计算损失
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()  # 累加损失
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")  # 打印每个 epoch 的损失

# 4. 文本生成
# 给定前4个字符，生成后续字符
model.eval()  # 切换到评估模式
input_str = "hell"  # 初始输入
input_idx = [stoi[ch] for ch in input_str]  # 转为索引
input_tensor = torch.tensor(input_idx).unsqueeze(0)  # (1, seq_len)
generated = list(input_str)  # 生成的字符序列

for _ in range(8):  # 生成 8 个字符
    logits = model(input_tensor)  # 前向传播
    next_idx = torch.argmax(logits, dim=-1).item()  # 取概率最大的下一个字符索引
    next_char = itos[next_idx]  # 转为字符
    generated.append(next_char)  # 加入生成序列
    input_tensor = torch.tensor([stoi[ch] for ch in generated[-seq_len:]]).unsqueeze(
        0
    )  # 更新输入

print("Transformer生成结果:", "".join(generated))  # 打印最终生成的文本
