import numpy as np

import torch
import torch.nn as nn

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange
from x_transformers import TransformerWrapper, Encoder, XTransformer

from .nn import normalization


class TextEncoder(nn.Module):
    def __init__(self,
        inner_dim,           # model dim (default = w_dim)
        depth = 2,
        head_dim = 64,
        num_tokens = 2500,
        max_seq_len = 64,
        rotary_pos_emb = True,
        ff_glu = True,
        use_scalenorm = True,
        use_rezero = False,
        use_encoder_decoder = False,
        decoder_sqrt_ntok = 32,
        encoder_kwargs = {},
        return_sequences=True
    ):
        super().__init__()

        head_dim = min(head_dim, inner_dim)

        assert inner_dim % head_dim == 0
        n_heads = inner_dim // head_dim

        self.use_encoder_decoder = use_encoder_decoder
        self.return_sequences = return_sequences

        if self.use_encoder_decoder:
            enc_kwargs = dict(
                depth = depth,
                heads = n_heads,
                rotary_pos_emb = rotary_pos_emb,
                ff_glu = ff_glu,
                use_scalenorm = use_scalenorm,
                use_rezero = use_rezero,
            )
            enc_kwargs = {k: encoder_kwargs.get(k, v) for k, v in enc_kwargs.items()}
            enc_kwargs = {'enc_' + k: v for k, v in enc_kwargs.items()}

            self.decoder_sqrt_ntok = decoder_sqrt_ntok
            self.dec_max_seq_len = decoder_sqrt_ntok ** 2

            self.model = XTransformer(
                enc_num_tokens = num_tokens,
                dec_num_tokens = 1,
                enc_max_seq_len = max_seq_len,
                dec_max_seq_len = self.dec_max_seq_len,
                dim = inner_dim,
                dec_depth = depth,
                dec_heads = n_heads,
                dec_rotary_pos_emb = rotary_pos_emb,
                dec_ff_glu = ff_glu,
                **enc_kwargs
            )
        else:
            self.model = TransformerWrapper(
                num_tokens = num_tokens,
                max_seq_len = max_seq_len,
                attn_layers = Encoder(
                    dim = inner_dim,
                    depth = depth,
                    heads = n_heads,
                    rotary_pos_emb = rotary_pos_emb,
                    ff_glu = ff_glu,
                    use_scalenorm = use_scalenorm,
                    use_rezero = use_rezero,
                )
            )
        if hasattr(self.model, "to_logits"):
            del self.model.to_logits

        # self.proj = nn.Linear(inner_dim, output_dim)

    def forward(self, tokens):
        if self.use_encoder_decoder:
            tgt = torch.zeros((tokens.shape[0], self.dec_max_seq_len), device=tokens.device, dtype=torch.int)
            enc = self.model.encoder(tokens, return_embeddings = True)
            out = self.model.decoder.net(tgt, context=enc, return_embeddings=True)
            out = rearrange(out, 'b (h w) c -> b h w c', h=self.decoder_sqrt_ntok)
            # out = self.proj(out)
            return out
        else:
            out = self.model(tokens, return_embeddings=True)
            if not self.return_sequences:
                out = out[:, 0, :]
            # out = self.proj(out)
            return out


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        text_dim=512,
        init_gain=1.,
        gain_scale=200.,
        resid=True,
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.text_dim = text_dim

        self.q = torch.nn.Linear(self.dim, self.dim, bias=False)
        self.kv = torch.nn.Linear(self.text_dim, 2*self.dim, bias=False)
        self.attn = torch.nn.MultiheadAttention(self.dim, self.heads, batch_first=True)

        self.src_ln = torch.nn.LayerNorm(self.text_dim)
        self.tgt_ln = normalization(self.dim)

        self.gain_scale = gain_scale
        self.gain = torch.nn.Parameter(torch.as_tensor(np.log(init_gain) / gain_scale))

        self.resid = resid

        torch.nn.init.orthogonal_(self.q.weight)
        torch.nn.init.orthogonal_(self.kv.weight)
        torch.nn.init.orthogonal_(self.attn.out_proj.weight)

    def forward(self, src, tgt):
        b, c, *spatial = tgt.shape
        tgt = tgt.reshape(b, c, -1)
        tgt = self.tgt_ln(tgt)
        tgt = tgt.transpose(1, 2)

        q = self.q(tgt)

        src = self.src_ln(src)
        kv = self.kv(src)

        k, v = kv.chunk(2, dim=-1)

        attn_output, attn_output_weights = self.attn(q, k, v)
        attn_output = (self.gain_scale * self.gain).exp() * attn_output
        attn_output = rearrange(attn_output, 'b (h w) c -> b c h w', h=spatial[0])

        if self.resid:
            tgt = tgt.reshape(b, c, *spatial)
            with torch.no_grad():
                norm_in = torch.linalg.norm(tgt).item()
                norm_add = torch.linalg.norm(attn_output).item()
            print(f"norm in {norm_in:.4f} | norm_add {norm_add:.4f}")
            return tgt + attn_output

        return attn_output
