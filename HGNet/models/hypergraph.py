
import torch
import torch.nn as nn


class MessageAgg(nn.Module):
    def __init__(self, agg_method="mean"):
        super().__init__()
        self.agg_method = agg_method

    def forward(self, X, path):
        """
            X: [n_node, dim]
            path: col(source) -> row(target)
        """
        X = torch.matmul(path, X)
        if self.agg_method == "mean":
            norm_out = 1 / torch.sum(path, dim=2, keepdim=True)
            norm_out[torch.isinf(norm_out)] = 0
            X = norm_out * X
            return X
        elif self.agg_method == "sum":
            pass
        return X


class HyPConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc = nn.Linear(c1, c2)
        self.v2e = MessageAgg(agg_method="mean")
        self.e2v = MessageAgg(agg_method="mean")

    def forward(self, x, H):
        x = self.fc(x)
        # v -> e
        E = self.v2e(x, H.transpose(1, 2).contiguous())
        # e -> v
        x = self.e2v(E, H)

        return x


class HyperComputeModule(nn.Module):
    def __init__(self, c1, c2, threshold, inter):
        super().__init__()
        self.threshold = threshold
        self.hgconv = HyPConv(c1, c2)
        self.bn = nn.BatchNorm1d(c2)
        self.act = nn.SiLU()
        self.inter = inter

    def forward(self, x):
        # b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        x = x.transpose(1, 2).contiguous()

        for _ in range(self.inter):
            feature = x.clone()  # 复制当前特征用于距离计算
            distance = torch.cdist(feature, feature)  # [B, H, H]
            hg = (distance < self.threshold).float().to(x.device).to(x.dtype)  # 构建邻接图
            x = self.hgconv(x, hg).to(x.device).to(x.dtype) + x  # 残差连接

        x = x.transpose(1, 2).contiguous()

        x = self.act(self.bn(x))

        return x
# def knn(x, k):
#     inner = -2*torch.matmul(x.transpose(2, 1), x)
#     xx = torch.sum(x**2, dim=1, keepdim=True)
#     pairwise_distance = -xx - inner - xx.transpose(2, 1)
#
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
#
#     return idx[:, :, :]

