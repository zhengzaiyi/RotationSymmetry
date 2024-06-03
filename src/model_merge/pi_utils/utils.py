import torch
from scipy.optimize import linear_sum_assignment

def get_submodule(m, name):
    ns = name.split(".")
    r = m
    for n in ns:
        r = getattr(r, n)
    return r


def get_QKVO(module, attn_pattern): # for attention
    Q = get_submodule(module, attn_pattern.query)
    K = get_submodule(module, attn_pattern.key)
    V = get_submodule(module, attn_pattern.value)
    O = get_submodule(module, attn_pattern.output)
        
    W_Q, B_Q = Q.weight.data, Q.bias.data
    W_K, B_K = K.weight.data, K.bias.data
    W_V, B_V = V.weight.data, V.bias.data
    
    W_O, B_O = O.weight.data, O.bias.data
    return {
        'W_Q': W_Q, 'B_Q': B_Q,
        'W_K': W_K, 'B_K': B_K,
        'W_V': W_V, 'B_V': B_V,
        'W_O': W_O, 'B_O': B_O
    }
    
def get_IO(module, ffn_pattern): # for FFN
    I = get_submodule(module, ffn_pattern.intermediate)
    O = get_submodule(module, ffn_pattern.output)
    W_I, B_I = I.weight, I.bias
    W_O_FFN, B_O_FFN = O.weight, O.bias
    return {
        'W_I': W_I, 'B_I': B_I,
        'W_O_FFN': W_O_FFN, 'B_O_FFN': B_O_FFN
    }
    
def Chunk_SVD_QK(
    local_param, anchor_param, 
    num_attention_heads=None,
    attention_head_size=64
):
    W_Q1, W_K1, B_Q1, B_K1 = local_param['W_Q'], local_param['W_K'], local_param['B_Q'], local_param['B_K']
    W_Q2, W_K2, B_Q2, B_K2 = anchor_param['W_Q'], anchor_param['W_K'], anchor_param['B_Q'], anchor_param['B_K']
    W_Q, W_K, B_Q, B_K = torch.zeros_like(W_Q1), torch.zeros_like(W_K1), torch.zeros_like(B_Q1), torch.zeros_like(B_K1)
    if not num_attention_heads:
        num_attention_heads = W_Q1.shape[0] // attention_head_size
    for i in range(num_attention_heads):
        start, end = i * attention_head_size, (i + 1) * attention_head_size
        W_Q1_, W_K1_, B_Q1_, B_K1_ = W_Q1[start:end], W_K1[start:end], B_Q1[start:end], B_K1[start:end]
        W_Q2_, W_K2_, B_Q2_, B_K2_ = W_Q2[start:end], W_K2[start:end], B_Q2[start:end], B_K2[start:end]
        summary = W_Q1_ @ W_Q2_.T + W_K1_ @ W_K2_.T + B_Q1.unsqueeze(0).T @ B_Q2.unsqueeze(0) + B_K1.unsqueeze(0).T @ B_K2.unsqueeze(0)
        U, _, VT = torch.linalg.svd(summary)
        V = VT.T
        P = V @ U.T
        P = P.T
        W_Q[start:end], W_K[start:end] = P.T @ W_Q1_, P.T @ W_K1_
        B_Q[start:end], B_K[start:end] = B_Q1_ @ P, B_K1_ @ P
    return {'W_Q': W_Q.detach(), 'B_Q': B_Q.detach(), 'W_K': W_K.detach(), 'B_K': B_K.detach()}

def Chunk_SVD_VO(
    local_param, anchor_param, 
    num_attention_heads=None,
    attention_head_size=64
):
    W_V1, W_O1, B_V1, B_O2 = local_param['W_V'], local_param['W_O'], local_param['B_V'], local_param['B_O']
    W_V2, W_O2, B_V2, B_O2 = anchor_param['W_V'], anchor_param['W_O'], anchor_param['B_V'], anchor_param['B_O']    
    W_V, W_O = torch.zeros_like(W_V2), torch.zeros_like(W_O2)
    B_V = torch.zeros_like(B_V2)
    if not num_attention_heads:
        num_attention_heads = W_V1.shape[0] // attention_head_size
    for i in range(num_attention_heads):
        start, end = i * attention_head_size, (i + 1) * attention_head_size
        W_V1_, W_O1_, B_V1_ = W_V1[start:end], W_O1[:, start:end], B_V1[start:end]
        W_V2_, W_O2_, B_V2_ = W_V2[start:end], W_O2[:, start:end], B_V2[start:end]
        summary = W_V1_ @ W_V2_.T + W_O1_.T @ W_O2_ + B_V1.unsqueeze(0).T @ B_V2.unsqueeze(0)
        
        U, _, VT = torch.linalg.svd(summary)
        V = VT.T
        P = V @ U.T
        P = P.T
        W_V[start:end], W_O[:, start:end] = P.T @ W_V1_, W_O1_ @ P
        B_V[start:end] = B_V1_ @ P
        
    return {'W_V': W_V.detach(), 'B_V': B_V.detach(), 'W_O': W_O.detach()}

def Permute_IO(
    local_param, anchor_param
):
    W_I1, W_O1, B_I1, B_O1 = local_param['W_I'], local_param['W_O_FFN'], local_param['B_I'], local_param['B_O_FFN']
    W_I2, W_O2, B_I2, B_O2 = anchor_param['W_I'], anchor_param['W_O_FFN'], anchor_param['B_I'], anchor_param['B_O_FFN']
    W_I, B_I, W_O = torch.zeros_like(W_I1), torch.zeros_like(B_I1), torch.zeros_like(W_O1)
    
    # shape W_I1: [3072, 768], W_O1: [768, 3072]
    cost = W_I1 @ W_I2.T + B_I1.unsqueeze(dim=0).T @ B_I2.unsqueeze(dim=0) + W_O1.T @ W_O2
    cost = cost.detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    P = torch.zeros_like(torch.tensor(cost))
    P[row_ind, col_ind] = 1
    P = P.to(W_I1.device)
    with torch.no_grad():
        W_I, B_I, W_O = P.T @ W_I1, B_I1 @ P, W_O1 @ P
    
    return {'W_I': W_I.detach(), 'B_I': B_I.detach(), 'W_O_FFN': W_O.detach()}