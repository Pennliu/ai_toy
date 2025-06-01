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
# 2. 模型定义
# =====================
HIDDEN_SIZE = 16
NUM_HEADS = 2
MAX_LEN = 16


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, max_len):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt):
        # src: [batch, src_len], tgt: [batch, tgt_len]
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        src_emb = self.src_embed(src) + self.pos_embed[:, :src_len, :]
        tgt_emb = self.tgt_embed(tgt) + self.pos_embed[:, :tgt_len, :]
        src_emb = src_emb.transpose(0, 1)  # [src_len, batch, d_model]
        tgt_emb = tgt_emb.transpose(0, 1)  # [tgt_len, batch, d_model]
        memory = self.encoder(src_emb)
        out = self.decoder(tgt_emb, memory)
        logits = self.fc(out)  # [tgt_len, batch, vocab]
        return logits.transpose(0, 1)  # [batch, tgt_len, vocab]


# =====================
# 3. 训练过程
# =====================
model = Seq2SeqTransformer(
    len(SRC_VOCAB), len(TGT_VOCAB), HIDDEN_SIZE, NUM_HEADS, MAX_LEN
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(300):
    model.train()
    src_tensor = torch.tensor(src_seq).unsqueeze(0)  # [1, 3]
    tgt_tensor = torch.tensor(tgt_seq[:-1]).unsqueeze(0)  # [1, 11]
    target = torch.tensor(tgt_seq[1:]).unsqueeze(0)  # [1, 11]
    logits = model(src_tensor, tgt_tensor)  # [1, 11, vocab]
    loss = criterion(logits.squeeze(0), target.squeeze(0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        pred_seq = []
        model.eval()
        src_tensor = torch.tensor(src_seq).unsqueeze(0)
        tgt_input = torch.tensor([[tgt2idx["<BOS>"]]])
        for _ in range(20):
            logits = model(src_tensor, tgt_input)
            next_token = logits[0, -1].argmax().item()
            ch = idx2tgt[next_token]
            if ch == "<EOS>":
                break
            pred_seq.append(ch)
            tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]])], dim=1)
        print(
            f"Epoch {
                epoch +
                1}, Loss: {
                loss.item():.4f}, 当前预测: {
                ''.join(pred_seq)}"
        )


# =====================
# 4. 推理（翻译生成）
# =====================


def translate(verbose=True):
    model.eval()
    src_tensor = torch.tensor(src_seq).unsqueeze(0)
    tgt_input = torch.tensor([[tgt2idx["<BOS>"]]])
    result = []
    for step in range(20):
        logits = model(src_tensor, tgt_input)
        next_token = logits[0, -1].argmax().item()
        ch = idx2tgt[next_token]
        if verbose:
            print(f"Step {step + 1}: 预测字符='{ch}'")
        if ch == "<EOS>":
            break
        result.append(ch)
        tgt_input = torch.cat([tgt_input, torch.tensor([[next_token]])], dim=1)
    return "".join(result)


print("\n推理详细过程：")
print("翻译结果：", translate(verbose=True))
