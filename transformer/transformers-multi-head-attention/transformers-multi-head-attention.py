import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    # 1. 线性投影
    q = Q @ W_q
    k = K @ W_k
    v = V @ W_v

    d_k = q.shape[-1] // num_heads
    d_v = v.shape[-1] // num_heads

    # 2. 拆头
    # [..., L, H*d] -> [..., L, H, d] -> [..., H, L, d]
    def split_heads(x):
        x = x.reshape(*x.shape[:-1], num_heads, x.shape[-1] // num_heads)
        return np.swapaxes(x, -3, -2)

    q = split_heads(q)
    k = split_heads(k)
    v = split_heads(v)

    # 3. 每个头计算注意力分数
    # [..., H, Lq, d_k] @ [..., H, d_k, Lk]
    scores = (q @ np.swapaxes(k, -1, -2)) / np.sqrt(d_k)

    # 4. 数值稳定的 softmax，沿 key 的维度归一化
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 5. 对 V 加权求和
    # [..., H, Lq, Lk] @ [..., H, Lk, d_v]
    out = weights @ v

    # 6. 合并各个头
    # [..., H, Lq, d_v] -> [..., Lq, H, d_v] -> [..., Lq, H*d_v]
    out = np.swapaxes(out, -3, -2)
    out = out.reshape(*out.shape[:-2], num_heads * d_v)

    # 7. 输出投影
    return out @ W_o