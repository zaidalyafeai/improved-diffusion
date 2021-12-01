import numpy as np

import torch
import torch.nn as nn

from axial_positional_embedding import AxialPositionalEmbedding
from einops import rearrange
from x_transformers import TransformerWrapper, Encoder, XTransformer
from x_transformers.x_transformers import AbsolutePositionalEmbedding

from .nn import normalization_1group, timestep_embedding, SiLU, AdaGN


def make_grad_mult_hook(mult, debug=False):
    def grad_mult_hook(grad):
        if debug:
            print(f"grad: {grad}")
        new_grad = mult * grad
        if debug:
            print(f"new_grad: {new_grad}")
        return new_grad
    return grad_mult_hook

def multiply_lr_via_hooks(m: nn.Module, mult: float, debug=False) -> nn.Module:
    m._lr_hook_handles = {}

    for n, p in m.named_parameters():
        handle = p.register_hook(make_grad_mult_hook(mult, debug=debug))
        m._lr_hook_handles[n] = handle

    return m


class LineEmbedding(nn.Module):
    def __init__(self, dim, line_sep_id=5, max_lines=32):
        super().__init__()
        self.line_sep_id = line_sep_id
        self.max_lines = max_lines

        self.scale = dim ** -0.5
        self.emb = nn.Embedding(max_lines, dim)

    def forward(self, x):
        n = (x == self.line_sep_id).to(torch.int).cumsum(dim=1).clamp(max=self.max_lines - 1)
        pos_emb = self.emb(n)
        return pos_emb * self.scale


class TextEncoder(nn.Module):
    def __init__(self,
        inner_dim,           # model dim (default = w_dim)
        depth = 2,
        head_dim = 64,
        num_tokens = 2500,
        max_seq_len = 64,
        rotary_pos_emb = False,
        ff_glu = False,
        use_scalenorm = True,
        use_rezero = False,
        use_encoder_decoder = False,
        decoder_sqrt_ntok = 32,
        encoder_kwargs = {},
        return_sequences=True,
        lr_mult=None,
        use_line_emb=True,
    ):
        super().__init__()

        head_dim = min(head_dim, inner_dim)

        assert inner_dim % head_dim == 0
        self.n_heads = inner_dim // head_dim

        self.use_encoder_decoder = use_encoder_decoder
        self.return_sequences = return_sequences
        self.use_line_emb = use_line_emb
        self.dim = inner_dim

        if self.use_encoder_decoder:
            enc_kwargs = dict(
                depth = depth,
                heads = self.n_heads,
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
                dec_heads = self.n_heads,
                dec_rotary_pos_emb = rotary_pos_emb,
                dec_ff_glu = ff_glu,
                **enc_kwargs
            )
        else:
            self.token_emb = nn.Embedding(num_tokens, inner_dim)
            self.pos_emb = AbsolutePositionalEmbedding(inner_dim, max_seq_len)
            if self.use_line_emb:
                self.line_emb = LineEmbedding(dim=inner_dim)
            self.model = Encoder(
                dim = inner_dim,
                depth = depth,
                heads = self.n_heads,
                rotary_pos_emb = rotary_pos_emb,
                ff_glu = ff_glu,
                use_scalenorm = use_scalenorm,
                use_rezero = use_rezero,
            )

            nn.init.kaiming_normal_(self.token_emb.weight)

        if hasattr(self.model, "to_logits"):
            del self.model.to_logits

        self.time_embed = nn.Linear(inner_dim, inner_dim)
        self.time_embed_scale = inner_dim ** -0.5
        # self.time_embed = nn.Sequential(
        #     nn.Linear(inner_dim, inner_dim),
        #     SiLU(),
        #     nn.Linear(inner_dim, inner_dim),
        # )

        # if lr_mult is not None:
        #     multiply_lr_via_hooks(self, lr_mult)

    def forward(self, tokens, timesteps=None):
        if self.use_encoder_decoder:
            tgt = torch.zeros((tokens.shape[0], self.dec_max_seq_len), device=tokens.device, dtype=torch.int)
            enc = self.model.encoder(tokens, return_embeddings = True)
            out = self.model.decoder.net(tgt, context=enc, return_embeddings=True)
            out = rearrange(out, 'b (h w) c -> b h w c', h=self.decoder_sqrt_ntok)
            # out = self.proj(out)
            return out
        else:
            x = tokens
            x = self.token_emb(x)
            # tok_norm = (x ** 2).sum().sqrt().item()
            pe = self.pos_emb(x)
            # pe_norm = (pe ** 2).sum().sqrt().item()
            x = x + pe
            if self.use_line_emb:
                le = self.line_emb(tokens)
                # le_norm = (le ** 2).sum().sqrt().item()
                x = x + le

            if timesteps is not None:
                emb = self.time_embed_scale * self.time_embed(timestep_embedding(timesteps, self.dim))
                emb = emb.unsqueeze(1).tile((1, x.shape[1], 1))
                # te_norm = (emb ** 2).sum().sqrt().item()
                x = x + emb

            # print((te_norm, tok_norm, pe_norm, le_norm))

            attn_mask = tokens != 0
            my_attn_mask = torch.tile(attn_mask.unsqueeze(1).unsqueeze(1), (self.n_heads, tokens.shape[1], 1))

            out = self.model(x, attn_mask=my_attn_mask)
            if not self.return_sequences:
                out = out[:, 0, :], attn_mask
            return out, attn_mask


class BetterMultiheadAttention(torch.nn.MultiheadAttention):
    def __init__(self, src_embed_dim, tgt_embed_dim, num_heads, dropout=0., batch_first=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(torch.nn.MultiheadAttention, self).__init__()
        self.src_embed_dim = src_embed_dim
        self.tgt_embed_dim = tgt_embed_dim
        self.embed_dim = self.src_embed_dim
        self.kdim = src_embed_dim
        self.vdim = src_embed_dim
        self._qkv_same_embed_dim = self.src_embed_dim == self.tgt_embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = src_embed_dim // num_heads
        assert self.head_dim * num_heads == self.src_embed_dim, "src_embed_dim must be divisible by num_heads"

        self.q = torch.nn.Linear(tgt_embed_dim, src_embed_dim, bias=False)
        self.k = torch.nn.Linear(src_embed_dim, src_embed_dim, bias=False)
        self.v = torch.nn.Linear(src_embed_dim, src_embed_dim, bias=False)

        self.scale = self.head_dim ** -0.5

        # self.fake_proj_weight = torch.nn.Parameter(torch.eye(src_embed_dim))
        # self.fake_proj_weight.requires_grad_(False)

        self.register_parameter('in_proj_weight', None)
        self.register_parameter('in_proj_bias', None)

        self.out_proj = torch.nn.modules.linear.NonDynamicallyQuantizableLinear(src_embed_dim, tgt_embed_dim, bias=False, **factory_kwargs)

        self.bias_k = self.bias_v = None
        self.add_zero_attn = False

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.q.weight)
        torch.nn.init.xavier_uniform_(self.k.weight)
        torch.nn.init.xavier_uniform_(self.v.weight)

    def forward(self, query, key, value,
                attn_mask=None,
                need_weights: bool = True):
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        query = self.q(query)
        key = self.k(key)
        value = self.v(value)

        query = self.scale * query

        fake_proj_weight = torch.eye(self.src_embed_dim, dtype=query.dtype, device=query.device)

        attn_output, attn_output_weights = torch.nn.functional.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            self.in_proj_weight, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.weight, self.out_proj.bias,
            training=self.training,
            key_padding_mask=None, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=True,
            q_proj_weight=fake_proj_weight, k_proj_weight=fake_proj_weight,
            v_proj_weight=fake_proj_weight)
        del fake_proj_weight

        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        emb_res,
        time_embed_dim,
        text_dim=512,
        init_gain=1.,
        gain_scale=200.,
        resid=True,
        lr_mult=None,
        needs_tgt_pos_emb=True,
        avoid_groupnorm=False,
        orth_init=False,
        q_t_emb=False,
        use_rezero=False,
        rezero_keeps_prenorm=False,
        use_layerscale=False,
        layerscale_init=1e-5,
    ):
        super().__init__()
        print(
            f"xattn: emb_res {emb_res} | dim {dim} | heads {heads} | avoid_groupnorm {avoid_groupnorm} | q_t_emb {q_t_emb} | use_rezero {use_rezero}"
        )
        self.dim = dim
        self.heads = heads
        self.text_dim = text_dim

        # self.q = torch.nn.Linear(self.dim, self.dim, bias=False)
        # self.kv = torch.nn.Linear(self.text_dim, 2*self.dim, bias=False)
        self.attn = BetterMultiheadAttention(self.text_dim, self.dim, self.heads, batch_first=True)

        self.avoid_groupnorm = avoid_groupnorm
        self.q_t_emb = q_t_emb
        self.use_rezero = use_rezero
        self.use_layerscale = use_layerscale
        self.no_prenorm = use_rezero and not rezero_keeps_prenorm

        if self.no_prenorm:
            self.src_ln = nn.Identity()
        else:
            self.src_ln = torch.nn.LayerNorm(self.text_dim)

        self.emb_res = emb_res
        self.tgt_pos_emb = None
        if needs_tgt_pos_emb:
            pos_emb_dim = self.dim // 2
            # pos emb in AdaGN
            if (not avoid_groupnorm) and self.q_t_emb:
                pos_emb_dim *= 2
            self.tgt_pos_emb = AxialPositionalEmbedding(
                dim=self.dim,
                axial_shape=(emb_res, emb_res),
                axial_dims=(pos_emb_dim, pos_emb_dim),
            )

        if self.q_t_emb:
            self.tgt_ln = AdaGN(
                emb_channels=time_embed_dim,
                out_channels=self.dim,
                num_groups=1,
                nonlin_in=True,  # TODO: does this matter?
                do_norm=not self.no_prenorm,
            )
        elif self.no_prenorm:
            self.tgt_ln = nn.Identity()
        elif avoid_groupnorm:
            self.tgt_ln = torch.nn.LayerNorm(self.dim)
        else:
            self.tgt_ln = normalization_1group(self.dim)

        self.gain_scale = gain_scale
        if self.use_layerscale:
            self.gain = torch.nn.Parameter(layerscale_init * torch.ones(self.dim))
        elif self.use_rezero:
            self.gain = torch.nn.Parameter(torch.zeros(1))
        else:
            self.gain = torch.nn.Parameter(torch.as_tensor(np.log(init_gain) / gain_scale))

        self.resid = resid

        if orth_init:
            torch.nn.init.orthogonal_(self.attn.q.weight)
            torch.nn.init.orthogonal_(self.attn.k.weight)
            torch.nn.init.orthogonal_(self.attn.v.weight)
            torch.nn.init.orthogonal_(self.attn.out_proj.weight)

        # if lr_mult is not None:
        #     multiply_lr_via_hooks(self, lr_mult)

    def effective_gain(self):
        g = self.gain_scale * self.gain
        if not (self.use_rezero or self.use_layerscale):
            g = g.exp()
        return g

    def forward(self, src, tgt, attn_mask=None, tgt_pos_embs=None, timestep_emb=None):
        def _to_b_hw_c(x, retdims=True):
            b, c, *spatial = x.shape
            xt = x.reshape(b, c, -1).transpose(1, 2)
            if retdims:
                return xt, b, c, spatial
            return xt

        def _to_b_c_h_w(x, spatial):
            return rearrange(x, 'b (h w) c -> b c h w', h=spatial[0])

        if tgt_pos_embs is None:
            tgt_pos_emb = self.tgt_pos_emb
        else:
            tgt_pos_emb = tgt_pos_embs[str(self.emb_res)]
        if tgt_pos_emb is None:
            raise ValueError('must pass tgt_pos_emb')

        if self.avoid_groupnorm:
            tgt_in, b, c, spatial = _to_b_hw_c(tgt)
            tgt_in = tgt_in + tgt_pos_emb(tgt_in)
            tgt_in = self.tgt_ln(tgt_in)
        elif self.q_t_emb:
            tgt_in = tgt

            b, c, *spatial = tgt_in.shape
            pos_emb = tgt_pos_emb(_to_b_hw_c(tgt_in, retdims=False))
            pos_emb = _to_b_c_h_w(pos_emb, spatial)

            tgt_in_norm1 = (tgt_in.float() ** 2).sum().sqrt().item()
            pos_emb_norm = (pos_emb.float() ** 2).sum().sqrt().item()
            ts_emb_norm = (timestep_emb.float() ** 2).sum().sqrt().item()

            print(("ins", tgt_in_norm1, ts_emb_norm, pos_emb_norm))

            tgt_in = self.tgt_ln(h=tgt_in, emb=timestep_emb, side_emb=pos_emb)
            tgt_in_norm2 = (tgt_in.float() ** 2).sum().sqrt().item()

            no_pos = self.tgt_ln(h=tgt_in, emb=timestep_emb)
            no_ts = self.tgt_ln(h=tgt_in, emb=torch.zeros_like(timestep_emb), side_emb=pos_emb)

            dpos_norm = ((tgt_in - no_pos).float() ** 2).sum().sqrt().item()
            dts_norm = ((tgt_in - no_ts).float() ** 2).sum().sqrt().item()

            print((tgt_in_norm2, dpos_norm, dts_norm))

            tgt_in, b, c, spatial = _to_b_hw_c(tgt_in)
        else:
            tgt_in = tgt
            tgt_in = self.tgt_ln(tgt_in)
            tgt_in, b, c, spatial = _to_b_hw_c(tgt_in)
            # pos emb after ln, so the GroupNorm doesn't avg it away
            tgt_in = tgt_in + tgt_pos_emb(tgt_in)

        # q = self.q(tgt_in)
        q = tgt_in

        src_in = self.src_ln(src)
        # kv = self.kv(src)
        # k, v = kv.chunk(2, dim=-1)
        k = src_in
        v = src_in

        my_attn_mask = None
        if attn_mask is not None:
            my_attn_mask = torch.tile(attn_mask.unsqueeze(1), (self.heads, q.shape[1], 1))
            my_attn_mask = (~my_attn_mask).to(q.dtype) * -10000.

        attn_output, attn_output_weights = self.attn(q, k, v, attn_mask=my_attn_mask)
        attn_output = attn_output * self.effective_gain()
        attn_output = _to_b_c_h_w(attn_output, spatial)

        if self.resid:
            return tgt + attn_output

        return attn_output
