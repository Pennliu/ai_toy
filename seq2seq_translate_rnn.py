import torch
import torch.nn as nn
import torch.optim as optim

# =====================
# 1. 词表和数据准备
# =====================
SRC_VOCAB = ["<BOS>", "<EOS>", "我", "爱", "你"]  # 中文字符表
TGT_VOCAB = [
    "<BOS>",
    "<EOS>",
    "I",
    " ",
    "l",
    "o",
    "v",
    "e",
    "y",
    "o",
    "u",
]  # 英文字符表

src2idx = {ch: i for i, ch in enumerate(SRC_VOCAB)}
tgt2idx = {ch: i for i, ch in enumerate(TGT_VOCAB)}
idx2tgt = {i: ch for ch, i in tgt2idx.items()}

# "我爱你" -> "I love you"
src_seq = [src2idx["我"], src2idx["爱"], src2idx["你"]]
tgt_seq = [
    tgt2idx["<BOS>"],
    tgt2idx["I"],
    tgt2idx[" "],
    tgt2idx["l"],
    tgt2idx["o"],
    tgt2idx["v"],
    tgt2idx["e"],
    tgt2idx[" "],
    tgt2idx["y"],
    tgt2idx["o"],
    tgt2idx["u"],
    tgt2idx["<EOS>"],
]

# =====================
# 4. 推理（翻译生成）
# =====================


def translate(verbose=True):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        src_tensor = torch.tensor(src_seq).unsqueeze(0)  # [1, 3]
        encoder_hidden = encoder(src_tensor)

        # 解码起始：<BOS>
        dec_input = torch.tensor([[tgt2idx["<BOS>"]]])  # [1, 1]
        hidden = encoder_hidden
        result = []
        for step in range(20):  # 最多生成20步，防止死循环
            # out: [1, 1, vocab], hidden: [1, 1, hidden]
            out, hidden = decoder(dec_input, hidden)
            pred = out.argmax(2)  # [1, 1]
            ch = idx2tgt[pred.item()]
            # 输出当前步信息
            if verbose:
                print(
                    f"Step {step + 1}: 预测字符='{ch}', "
                    f"隐藏状态前5位={hidden[0, 0, :5].cpu().numpy()}"
                )
            if ch == "<EOS>":
                break
            result.append(ch)
            dec_input = pred  # 上一步输出作为下一步输入
        return "".join(result)


# =====================
# 2. 模型定义
# =====================
HIDDEN_SIZE = 16


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(SRC_VOCAB), HIDDEN_SIZE)
        self.rnn = nn.RNN(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embed(x)  # [batch, seq_len, hidden]
        # outputs: [batch, seq_len, hidden], hidden: [1, batch, hidden]
        outputs, hidden = self.rnn(x)
        return hidden  # 只返回最后的隐藏状态


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(TGT_VOCAB), HIDDEN_SIZE)
        self.rnn = nn.RNN(HIDDEN_SIZE, HIDDEN_SIZE, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, len(TGT_VOCAB))

    def forward(self, x, hidden):
        # x: [batch, seq_len]
        x = self.embed(x)  # [batch, seq_len, hidden]
        # outputs: [batch, seq_len, hidden]
        outputs, hidden = self.rnn(x, hidden)
        logits = self.fc(outputs)  # [batch, seq_len, vocab]
        return logits, hidden


# =====================
# 3. 训练过程（增加每50轮输出当前预测序列）
# =====================
encoder = Encoder()
decoder = Decoder()
criterion = nn.CrossEntropyLoss()
enc_opt = optim.Adam(encoder.parameters(), lr=0.01)
dec_opt = optim.Adam(decoder.parameters(), lr=0.01)

for epoch in range(300):
    # 1. 编码器：输入中文序列
    src_tensor = torch.tensor(src_seq).unsqueeze(0)  # [1, 3]
    encoder_hidden = encoder(src_tensor)  # [1, 1, hidden]

    # 2. 解码器：输入目标序列（去掉最后一个<EOS>），目标是下一个字符
    tgt_tensor = torch.tensor(tgt_seq[:-1]).unsqueeze(0)  # [1, 11]
    target = torch.tensor(tgt_seq[1:]).unsqueeze(0)  # [1, 11]
    # dec_out: [1, 11, vocab]
    dec_out, _ = decoder(tgt_tensor, encoder_hidden)
    dec_out = dec_out.squeeze(0)  # [11, vocab]
    loss = criterion(dec_out, target.squeeze(0))

    # 3. 反向传播和优化
    enc_opt.zero_grad()
    dec_opt.zero_grad()
    loss.backward()
    enc_opt.step()
    dec_opt.step()

    if (epoch + 1) % 50 == 0:
        pred_seq = translate(verbose=False)
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, 当前预测: {pred_seq}")

print("\n推理详细过程：")
print("翻译结果：", translate(verbose=True))
