import torch
import torch.nn as nn

class TinyAttention(nn.Module):
    def __init__(self, d_out, d_attn):
        super(TinyAttention, self).__init__()
        self.kqv = nn.Linear(d_out, 3*d_attn)
        if(d_attn != d_out):
            self.proj = nn.Linear(d_attn, d_out)
        else:
            self.proj = None
            
    def forward(self, x):
        kqv = self.kqv(x)
        key, query, value = kqv.chunk(3, dim=-1)
        soft = nn.Softmax(dim=-1)((query @ key.permute(0, 2, 1)) / torch.sqrt(torch.tensor(query.shape[-1]).float()))
        head = soft @ value
        if(self.proj is not None):
            head = self.proj(head)
        return head

class SpatialGatingUnit(nn.Module):
    def __init__(self, hidden_dim, len_seq, split, toeplitz, tn_attn, d_attn):
        super(SpatialGatingUnit, self).__init__()
        self.split = split
        if split:
            self.norm = nn.LayerNorm(hidden_dim // 2)
        else:
            self.norm = nn.LayerNorm(hidden_dim)
        
        self.len_seq = len_seq
        self.toeplitz = toeplitz
        if(toeplitz):
            zeros, ones = torch.zeros(2*len_seq - 1), torch.ones(len_seq)
            self.w_vector, self.bias = nn.Parameter(zeros), nn.Parameter(ones)
        else:
            self.proj = nn.Linear(len_seq, len_seq)
            torch.nn.init.zeros_(self.proj.weight)
            self.proj.bias.data.fill_(1)
        
        if(tn_attn):
            if split:
                self.tiny_attn = TinyAttention(hidden_dim//2, d_attn)
            else:
                self.tiny_attn = TinyAttention(hidden_dim, d_attn)
        else:
            self.tiny_attn = None
    
    def get_toeplitz(self):
        r = self.w_vector.shape[0] // 2
        t = torch.nn.functional.pad(self.w_vector, (0, self.len_seq), "constant", 0)
        t = t.repeat(self.len_seq)
        t = t[: -self.len_seq]
        t = torch.reshape(t, [self.len_seq, self.len_seq + self.w_vector.shape[0] - 1])
        return t[:, r:-r]
    
    def forward(self, x):
        if self.split:
            u, v = x.chunk(2, dim=-1)
        else:
            u, v = x, x
        
        if(self.tiny_attn is not None):
            head = self.tiny_attn(v)
        else:
            head = 0
        
        v = self.norm(v)
        v = v.permute(0, 2, 1)
        if(self.toeplitz):
            w_toeplitz = self.get_toeplitz()
            v = v @ w_toeplitz + self.bias
        else:
            v = self.proj(v)
        v = v.permute(0, 2, 1) + head
        return u * v

class MLP_BLock(nn.Module):
    def __init__(self, hidden_dim: int, shape, split, toeplitz, tn_attn, d_attn):
        super(MLP_BLock, self).__init__()
        self.proj_1 = nn.Sequential(
            nn.Linear(shape[1], hidden_dim, bias=False),
            nn.GELU()
        )
        if split:
            self.proj_2 = nn.Linear(hidden_dim // 2, shape[1])
        else:
            self.proj_2 = nn.Linear(hidden_dim, shape[1])
        
        self.sgu = SpatialGatingUnit(hidden_dim, shape[0], split, toeplitz, tn_attn, d_attn)
        
        self.norm = nn.LayerNorm(shape[1])
        self.norm1 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        out = self.norm(x)
        out = self.proj_1(out)
        out = self.sgu(out)
        out = self.proj_2(out)
        out = out + x
        return out

class MlpNN(nn.Module):
    """Mixer architecture."""
    def __init__(self, MLP_BLock, vocab_size, embed_dim, num_blocks, hidden_dim, len_seq, num_classes=2, split=True,
                toeplitz=False, tn_attn=False, d_attn=64):
        super(MlpNN, self).__init__()
        self.hidden_dim = hidden_dim # C
        self.len_seq = len_seq # S

        block_sequence = []
        for i in range(num_blocks):
            block_sequence.append(MLP_BLock(hidden_dim, (len_seq, embed_dim), split, toeplitz, tn_attn, d_attn))
        self.block_sequence = nn.Sequential(*block_sequence)

        self.fc = nn.Linear(embed_dim, num_classes)
        if(embed_dim != 768):
            self.proj = nn.Linear(768, embed_dim)
        else:
            self.proj = None
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        if (self.proj is not None):
            x = self.proj(x)
        x = self.block_sequence(x)
        x = self.norm(x)
        x = torch.mean(x, axis=1)
        x = self.fc(x)
        return x

def gMLP(vocab_size, embed_dim, num_blocks, hidden_dim, len_seq, split=True, toeplitz=False, tn_attn=False, d_attn=64):
    return MlpNN(MLP_BLock, vocab_size=vocab_size, embed_dim=embed_dim, 
                 num_blocks=num_blocks, hidden_dim=hidden_dim, len_seq=len_seq, split=split,
                toeplitz=toeplitz, tn_attn=tn_attn, d_attn=d_attn)
