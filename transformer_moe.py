import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 传统 Transformer 的前馈网络 (FFN) ---


class TraditionalFeedForward(nn.Module):
    """
    传统 Transformer 中的前馈网络 (FFN)。
    每个输入 token 都会通过这两个线性层。
    """

    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)  # 第一个线性层
        self.gelu = nn.GELU()  # 激活函数
        self.linear2 = nn.Linear(ff_dim, embed_dim)  # 第二个线性层

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)

        # 所有输入 token 都会经过线性层1
        intermediate = self.linear1(x)
        intermediate = self.gelu(intermediate)

        # 所有输入 token 都会经过线性层2
        output = self.linear2(intermediate)

        return output


# --- 2. MoE Transformer 的 MoE 层 ---


class Expert(nn.Module):
    """
    一个"专家"网络，实际上就是一个小的 FFN。
    它与 TraditionalFeedForward 的结构相同，但作为 MoE 的一个组件。
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
    Mixture-of-Experts (MoE) 层。
    包含多个专家 FFN 和一个门控网络来选择专家。
    """

    def __init__(self, embed_dim, ff_dim, num_experts, top_k):
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot be greater than num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        # 创建多个专家实例
        self.experts = nn.ModuleList(
            [Expert(embed_dim, ff_dim) for _ in range(num_experts)]
        )
        # 门控网络，用于为每个 token 预测其应路由到哪个专家
        self.gate = nn.Linear(embed_dim, num_experts)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)

        batch_size, seq_len, embed_dim = x.shape

        # 将输入展平，以便门控网络可以处理每个 token
        flat_x = x.view(-1, embed_dim)  # (batch_size * seq_len, embed_dim)

        # 1. 门控网络计算每个专家的得分
        gate_logits = self.gate(flat_x)  # (batch_size * seq_len, num_experts)

        # 2. 对得分应用 softmax 得到路由概率
        # (batch_size * seq_len, num_experts)
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 3. 选择 Top-K 专家
        # top_k 会返回前 top_k 个值和它们的索引
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)

        # 为了正确加权，将 top_k_probs 归一化，使其和为1
        # 避免除以零，添加一个小的 epsilon
        normalizing_constant = top_k_probs.sum(dim=-1, keepdim=True) + 1e-9
        # (batch_size * seq_len, top_k)
        top_k_weights = top_k_probs / normalizing_constant

        # 4. 初始化输出张量
        output = torch.zeros_like(flat_x)  # (batch_size * seq_len, embed_dim)

        # 5. 将输入路由到选定的专家，并进行加权求和
        # 遍历每个 (样本, token) 组合
        for i in range(batch_size * seq_len):
            input_token_i = flat_x[i]  # 当前 token 的嵌入向量

            # 为当前 token 选中的专家索引和权重
            selected_expert_indices = top_k_indices[i]
            selected_expert_weights = top_k_weights[i]

            # 遍历选中的 top_k 个专家
            for k_rank in range(self.top_k):
                expert_idx = selected_expert_indices[k_rank].item()  # 实际专家的索引
                weight = selected_expert_weights[k_rank]  # 对应的权重

                # 激活并计算选中的专家
                # 注意：这里对单个 token 进行操作，所以需要 unsqueeze(0) 增加 batch 维度
                expert_output = self.experts[expert_idx](
                    input_token_i.unsqueeze(0)
                )  # (1, embed_dim)

                # 将专家输出乘以对应的权重并累加到总输出
                output[i] += expert_output.squeeze(0) * weight

        # 将输出形状恢复到原始的 (batch_size, seq_len, embed_dim)
        return output.view(batch_size, seq_len, embed_dim)


# --- 演示和对比 ---


# 模型参数
embed_dim = 768  # 嵌入维度
ff_dim = 3072  # FFN 的中间维度 (传统上是 embed_dim 的 4 倍)
num_experts = 8  # MoE 中专家的总数量
top_k = 2  # MoE 中每个 token 激活的专家数量

batch_size = 2  # 批次大小
seq_len = 10  # 序列长度

# 模拟输入数据
dummy_input = torch.randn(batch_size, seq_len, embed_dim)  # 随机生成输入

print("--- 传统 Transformer FFN vs. MoE Layer 对比 ---")

# --- 传统 FFN 演示 ---
print("\n=== 传统 Transformer FFN ===")
traditional_ffn = TraditionalFeedForward(embed_dim, ff_dim)
print(
    f"传统 FFN 参数总量: {sum(p.numel() for p in traditional_ffn.parameters()):,} 个")

output_traditional = traditional_ffn(dummy_input)
print(f"输入形状: {dummy_input.shape}")
print(f"输出形状: {output_traditional.shape}")
print("特点: 每次前向传播，FFN 的所有参数都会被激活和计算。")

# --- MoE Layer 演示 ---
print("\n=== MoE Layer ===")
moe_layer = MoELayer(embed_dim, ff_dim, num_experts, top_k)
print(
    f"MoE Layer 总参数总量 (包含所有专家和门控网络): "
    f"{sum(p.numel() for p in moe_layer.parameters()):,} 个"
)

output_moe = moe_layer(dummy_input)
print(f"输入形状: {dummy_input.shape}")
print(f"输出形状: {output_moe.shape}")
print(f"特点: MoE 层共有 {num_experts} 个专家。")
print(f"每个输入 token 仅激活其中 {top_k} 个专家。")
print(
    f"这意味着在每个前向传播步中，只有总参数的约 ({top_k}/{num_experts}) * 100% = ({
        top_k / num_experts:.2f}) * 100% 的专家参数被激活。"
)
print(
    "尽管 MoE 的总参数量可能远大于传统 FFN，但其稀疏激活特性使其在推理时实际计算量更低。"
)
