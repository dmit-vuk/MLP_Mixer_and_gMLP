import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

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

class MixerBlock(nn.Module):
    """Mixer block layer."""
    def __init__(self, tokens_mlp_dim: int, channels_mlp_dim: int, shape, tn_attn=False, d_attn=64):
        super(MixerBlock, self).__init__()
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim

        self.norm1 = nn.LayerNorm(shape[1])
        self.norm2 = nn.LayerNorm(shape[1])

        self.mlp_channel = nn.Sequential(
            nn.Linear(shape[1], channels_mlp_dim),
            nn.GELU(),
            nn.Linear(channels_mlp_dim, shape[1])
        )
        self.mlp_token = nn.Sequential(
            Rearrange('b n d -> b d n'),
            nn.Linear(shape[0], tokens_mlp_dim),
            nn.GELU(),
            nn.Linear(tokens_mlp_dim, shape[0]),
            Rearrange('b n d -> b d n'),
        )
        
        if(tn_attn):
            self.tiny_attn = TinyAttention(shape[1], d_attn)
        else:
            self.tiny_attn = None
  
    def forward(self, x):
        out = self.norm1(x)
        if self.tiny_attn is not None:
            head = self.tiny_attn(out)
        else:
            head = 0
        out = self.mlp_token(out)
        x = out + x + head
        out = self.norm2(x)
        out = self.mlp_channel(out)
        out = out + x
        return out

class MlpMixer(nn.Module):
    """Mixer architecture."""
    def __init__(self, MixerBlock, patches, num_blocks, hidden_dim, num_patches, tokens_mlp_dim, channels_mlp_dim, num_classes=10,
                tn_attn=False, d_attn=64):
        super(MlpMixer, self).__init__()

        self.hidden_dim = hidden_dim # C
        self.num_patches = num_patches # S

        block_sequence = []
        for i in range(num_blocks):
            block_sequence.append(MixerBlock(tokens_mlp_dim, channels_mlp_dim, (num_patches, hidden_dim), tn_attn=tn_attn, d_attn=d_attn))
        self.block_sequence = nn.Sequential(*block_sequence)

        self.head = nn.Linear(hidden_dim, num_classes)
        self.norm = nn.LayerNorm(hidden_dim)
        self.conv = nn.Conv2d(3, hidden_dim, kernel_size=patches, stride=patches)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = Rearrange('n c h w -> n (h w) c')(x)
        x = self.block_sequence(x)
        x = self.norm(x)
        x = torch.mean(x, axis=1)
        x = self.head(x)
        return x

def Mixer(patches, num_blocks, hidden_dim, num_patches, tokens_mlp_dim, channels_mlp_dim, tn_attn=False, d_attn=64):
    return MlpMixer(MixerBlock, patches=patches, num_blocks=num_blocks, hidden_dim=hidden_dim, tn_attn=tn_attn, d_attn=d_attn,
                    num_patches=num_patches, tokens_mlp_dim=tokens_mlp_dim, channels_mlp_dim=channels_mlp_dim)
