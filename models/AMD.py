import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MDM(nn.Module):
    """
    Multi-scale Decomposition Module from AMD.

    This is copied from AMD-main/models/common.py but without RevIN
    so that HDN handles normalization instead.
    """

    def __init__(self, input_shape, k: int = 3, c: int = 2, layernorm: bool = True):
        """
        Args:
            input_shape: (seq_len, feature_num)
            k: number of downsampling scales (0 disables the module)
            c: scale factor between levels
            layernorm: whether to apply BatchNorm-based layer normalization
        """
        super().__init__()
        self.seq_len = input_shape[0]
        self.k = k
        if self.k > 0:
            # k_list contains the pooling kernel/stride sizes for each scale
            self.k_list = [c ** i for i in range(k, 0, -1)]
            self.avg_pools = nn.ModuleList(
                [nn.AvgPool1d(kernel_size=kk, stride=kk) for kk in self.k_list]
            )
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.seq_len // kk, self.seq_len // kk),
                        nn.GELU(),
                        nn.Linear(self.seq_len // kk, self.seq_len * c // kk),
                    )
                    for kk in self.k_list
                ]
            )
        self.layernorm = layernorm
        if self.layernorm:
            # Flatten seq_len * feature_dim and apply BN1d
            self.norm = nn.BatchNorm1d(input_shape[0] * input_shape[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, feature_num, seq_len]
        Returns:
            Tensor with same shape after multi-scale mixing.
        """
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)
        if self.k == 0:
            return x

        # x: [batch, feature_num, seq_len]
        sample_x = []
        for i, kk in enumerate(self.k_list):
            sample_x.append(self.avg_pools[i](x))
        sample_x.append(x)
        n = len(sample_x)

        # Progressive residual upsampling across scales
        for i in range(n - 1):
            tmp = self.linears[i](sample_x[i])
            sample_x[i + 1] = torch.add(sample_x[i + 1], tmp, alpha=1.0)

        # [batch, feature_num, seq_len]
        return sample_x[n - 1]


class DDI(nn.Module):
    """
    Dynamic Decomposition Interaction block from AMD.

    Copied from AMD-main/models/common.py (RevIN removed).
    """

    def __init__(
        self,
        input_shape,
        dropout: float = 0.2,
        patch: int = 12,
        alpha: float = 0.0,
        layernorm: bool = True,
    ):
        """
        Args:
            input_shape: (seq_len, feature_num)
            dropout: dropout rate
            patch: patch length along time dimension
            alpha: strength of channel-wise FFN refinement (0 disables it)
            layernorm: whether to apply BatchNorm-based layer normalization
        """
        super().__init__()
        # input_shape[0] = seq_len, input_shape[1] = feature_num
        self.input_shape = input_shape
        if alpha > 0.0:
            self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
            self.fc_block = nn.Sequential(
                nn.Linear(self.input_shape[-1], self.ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_dim, self.input_shape[-1]),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.n_history = 1
        self.alpha = alpha
        self.patch = patch

        self.layernorm = layernorm
        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        if self.alpha > 0.0:
            self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])

        self.agg = nn.Linear(self.n_history * self.patch, self.patch)
        self.dropout_t = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, feature_num, seq_len]
        Returns:
            Tensor with same shape after local history aggregation.
        """
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        output = torch.zeros_like(x)
        output[:, :, : self.n_history * self.patch] = x[
            :, :, : self.n_history * self.patch
        ].clone()
        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            # input: [batch, feature_num, n_history * patch]
            inp = output[:, :, i - self.n_history * self.patch : i]
            inp = self.norm1(torch.flatten(inp, 1, -1)).reshape(inp.shape)
            # aggregation: [batch, feature_num, patch]
            inp = F.gelu(self.agg(inp))  # n_history * patch -> patch
            inp = self.dropout_t(inp)
            tmp = inp + x[:, :, i : i + self.patch]
            res = tmp

            if self.alpha > 0.0:
                tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
                tmp = torch.transpose(tmp, 1, 2)  # [batch, patch, feature_num]
                tmp = self.fc_block(tmp)
                tmp = torch.transpose(tmp, 1, 2)  # [batch, feature_num, patch]
            output[:, :, i : i + self.patch] = res + self.alpha * tmp

        # [batch, feature_num, seq_len]
        return output


class TopKGating(nn.Module):
    """
    Top-k gating network from AMD-main/models/tsmoe.py.
    """

    def __init__(self, input_dim: int, num_experts: int, top_k: int = 2, noise_epsilon: float = 1e-5):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.top_k = top_k
        self.noise_epsilon = noise_epsilon
        self.num_experts = num_experts
        self.w_noise = nn.Parameter(torch.zeros(num_experts, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)

    def decompostion_tp(self, x: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
        # x: [batch, num_experts]
        output = torch.zeros_like(x)
        kth_largest_val, _ = torch.kthvalue(x, self.num_experts - self.top_k + 1)
        kth_largest_mat = kth_largest_val.unsqueeze(1).expand(-1, self.num_experts)
        mask = x < kth_largest_mat
        x = self.softmax(x)
        output[mask] = alpha * torch.log(x[mask] + 1)
        output[~mask] = alpha * (torch.exp(x[~mask]) - 1)
        # Ablation Spare MoE:
        # output[mask] = 0
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_dim]
        x = self.gate(x)
        clean_logits = x

        if self.training:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.decompostion_tp(logits)
        gates = self.softmax(logits)
        return gates


class Expert(nn.Module):
    """
    Single expert MLP used inside AMS.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AMS(nn.Module):
    """
    Adaptive Mixture-of-Experts head from AMD-main/models/tsmoe.py.
    """

    def __init__(
        self,
        input_shape,
        pred_len: int,
        ff_dim: int = 2048,
        dropout: float = 0.2,
        loss_coef: float = 1.0,
        num_experts: int = 4,
        top_k: int = 2,
    ):
        """
        Args:
            input_shape: (seq_len, feature_num)
            pred_len: length of prediction horizon
        """
        super().__init__()
        # input_shape[0] = seq_len, input_shape[1] = feature_num
        self.num_experts = num_experts
        self.top_k = top_k
        self.pred_len = pred_len

        self.gating = TopKGating(input_shape[0], num_experts, top_k)
        self.experts = nn.ModuleList(
            [Expert(input_shape[0], pred_len, hidden_dim=ff_dim, dropout=dropout) for _ in range(num_experts)]
        )
        self.loss_coef = loss_coef
        assert self.top_k <= self.num_experts

    @staticmethod
    def cv_squared(x: torch.Tensor) -> torch.Tensor:
        eps = 1e-10
        # if only num_experts = 1
        if x.shape[0] == 1:
            return torch.tensor([0.0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor):
        """
        Args:
            x: [batch, feature_num, seq_len]
            time_embedding: [batch, feature_num, seq_len]
        Returns:
            output: [batch, feature_num, pred_len]
            moe_loss: scalar tensor (not used by HDN training, but kept for completeness)
        """
        batch_size = x.shape[0]
        feature_num = x.shape[1]

        # [feature_num, batch, seq_len]
        x = torch.transpose(x, 0, 1)
        time_embedding = torch.transpose(time_embedding, 0, 1)

        output = torch.zeros(feature_num, batch_size, self.pred_len, device=x.device, dtype=x.dtype)
        loss = 0.0

        for i in range(feature_num):
            inp = x[i]  # [batch, seq_len]
            time_info = time_embedding[i]
            gates = self.gating(time_info)  # [batch, num_experts]

            # expert_outputs: [num_experts, batch, pred_len]
            expert_outputs = torch.zeros(self.num_experts, batch_size, self.pred_len, device=x.device, dtype=x.dtype)
            for j in range(self.num_experts):
                expert_outputs[j, :, :] = self.experts[j](inp)
            expert_outputs = torch.transpose(expert_outputs, 0, 1)  # [batch, num_experts, pred_len]

            gates_expanded = gates.unsqueeze(-1).expand(-1, -1, self.pred_len)
            batch_output = (gates_expanded * expert_outputs).sum(1)  # [batch, pred_len]
            output[i, :, :] = batch_output

            importance = gates.sum(0)
            loss = loss + self.loss_coef * self.cv_squared(importance)

        # [feature_num, batch, pred_len] -> [batch, feature_num, pred_len]
        output = torch.transpose(output, 0, 1)
        return output, loss


class Model(nn.Module):
    """
    AMD backbone adapted for the HDN framework.

    Differences from the original AMD implementation:
      - RevIN-based normalization is removed; HDN's Statistics_prediction
        module handles normalization instead.
      - The forward signature is (x, k) to be compatible with
        Exp_HDN.backbone_forward for 'Linear' style models; `k`
        (period index) is unused but kept for API consistency.
      - The auxiliary MoE loss is computed but not added to the training
        objective inside this module; if desired, it can be accessed via
        `self.latest_moe_loss` after a forward pass.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        seq_len = configs.seq_len
        feature_num = configs.enc_in if configs.features == 'M' else 1
        input_shape = (seq_len, feature_num)

        pred_len = configs.pred_len

        # AMD-specific hyperparameters (with defaults matching AMD-main)
        n_block = getattr(configs, "amd_n_block", 1)
        alpha = getattr(configs, "amd_alpha", 0.0)
        mix_layer_num = getattr(configs, "amd_mix_layer_num", 3)
        mix_layer_scale = getattr(configs, "amd_mix_layer_scale", 2)
        patch = getattr(configs, "amd_patch", 16)
        dropout = getattr(configs, "amd_dropout", 0.1)
        layernorm = getattr(configs, "amd_layernorm", True)

        self.fc_blocks = []
        
        self.pastmixing = MDM(input_shape, k=mix_layer_num, c=mix_layer_scale, layernorm=layernorm)
        # 1. 必须首先初始化为 nn.ModuleList，而不是 []
        self.fc_blocks = nn.ModuleList() 

        # 2. 循环构建
        for i in range(configs.top_k): 
            # 这里构建每一组的 blocks
            k_branch = nn.ModuleList([
                DDI(
                    input_shape, 
                    dropout=dropout, 
                    # 注意：如果 patch 需要 int，建议加上 .item()，或者确保 DDI 内部处理了 Tensor
                    patch=configs.period_list[i].cpu(), 
                    alpha=alpha, 
                    layernorm=layernorm
                )
                for _ in range(n_block)
            ])
            
            # 3. 将这一组 ModuleList 添加到外层的 ModuleList 中
            self.fc_blocks.append(k_branch)

        
        self.moe = AMS(
            input_shape,
            pred_len,
            ff_dim=2048,
            dropout=dropout,
            num_experts=8,
            top_k=2,
        )

        # Expose latest MoE loss in case the training loop wants to use it
        self.latest_moe_loss = None

    def forward(self, x: torch.Tensor, k: int = None, return_moe_loss: bool = False):
        """
        Args:
            x: [batch, seq_len, feature_num]
            k: period index used by HDN (ignored here, kept for API compatibility)
            return_moe_loss: whether to also return the MoE auxiliary loss
        Returns:
            If return_moe_loss is False:
                outputs: [batch, pred_len, feature_num]
            If return_moe_loss is True:
                (outputs, moe_loss)
        """
        # [batch, seq_len, feature_num] -> [batch, feature_num, seq_len]
        x = torch.transpose(x, 1, 2)
        
        time_embedding = self.pastmixing(x)

        for fc_block in self.fc_blocks[k]:
            x = fc_block(x)

        # Mixture of experts maps seq_len -> pred_len
        x, moe_loss = self.moe(x, time_embedding)  # x: [batch, feature_num, pred_len]
        self.latest_moe_loss = moe_loss

        # [batch, feature_num, pred_len] -> [batch, pred_len, feature_num]
        x = torch.transpose(x, 1, 2)
        if return_moe_loss:
            return x, moe_loss
        return x
