# modules/plot_bridge.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.plot_impl.slot_model import SlotAttention_LearnableSlots, MlpDecoder
from modules.plot_impl import objectives as plot_obj
import math

class SlotAdapter(nn.Module):
    """
    调用 PLOT 的 SlotAttention 和 objectives.compute_nce，
    不改动 PLOT 源码；按帧逐一计算，再对帧（以及可选的多文本对）做平均。
    """
    def __init__(self, dim, num_slots=8, num_iters=5):
        super().__init__()
        
        self.dim = dim
        self.num_slots = num_slots
        self.num_iters = num_iters

        self.slots = nn.Parameter(torch.randn(num_slots, dim) / (dim ** 0.5))
        nn.init.orthogonal_(self.slots)

        # 分别用于视觉/文本的 PSA（结构相同，参数独立）
        self.slot_v = SlotAttention_LearnableSlots(num_slots, dim, iters=num_iters)
        self.slot_t = SlotAttention_LearnableSlots(num_slots, dim, iters=num_iters)

        # 文本侧槽权重（与 PLOT 一致：text_global -> num_slots）
        self.text_slot_classifier = nn.Sequential(
            nn.Linear(dim, dim // 8), nn.ReLU(inplace=True), nn.Linear(dim // 8, num_slots)
        )

        self.recon_dec_v = MlpDecoder(7, 7, slot_dim=dim, feat_dim=dim)
        self.recon_dec_t = MlpDecoder(1, 76, slot_dim=dim, feat_dim=dim)

    def forward(self, vid_tokens_all, txt_tokens_all, txt_mask_all, txt_global_all, logit_scale, pid,
                use_metric=True, use_recon=False, recon_weight=0.01, pos_across_frames=False):
        """
        vid_tokens_all: [B, T, Lv, D]   (可含 CLS)
        txt_tokens_all: [B, P, Lt, D]   (可含 CLS)
        txt_mask_all  : [B, P, Lt] or None
        txt_global_all: [B, P, D]
        pid           : [B]    （无 ID 时可传 arange(B)）
        pos_across_frames: 若 True，同一视频不同帧也视为正样本；若 False，仅逐帧配对为正（与“逐帧求 NCE 再平均”等价）
        """
        B, T = vid_tokens_all.shape[:2]
        P = txt_tokens_all.shape[1]

        # ----- 预处理：去 CLS -----
        v_tok = vid_tokens_all[..., 1:, :]                      # [B, T, Lv-1, D]
        t_tok = txt_tokens_all[..., 1:, :]                      # [B, P, Lt-1, D]
        t_msk = (txt_mask_all[..., 1:] if txt_mask_all is not None else None)

        # ----- 文本侧 SlotAttention：一次算完所有句子 -----
        t_tok_flat  = t_tok.reshape(B*P, t_tok.size(-2), t_tok.size(-1))      # [B*P, Lt-1, D]
        t_msk_flat  = t_msk.reshape(B*P, t_msk.size(-1)) if t_msk is not None else None
        t_list, t_attn = self.slot_t(self.slots, t_tok_flat, mask=t_msk_flat) # t_attn: [B*P, Nt, S]
        t_local = t_list[-1].reshape(B, P, -1, t_list[-1].size(-1))           # [B, P, S, D]
        t_attn  = t_attn.reshape(B, P, t_attn.size(1), t_attn.size(2))        # [B, P, Nt, S]
        part_w  = self.text_slot_classifier(txt_global_all)                    # [B, P, S]

        # ----- 视频侧 SlotAttention：一次算完所有帧 -----
        v_tok_flat = v_tok.reshape(B*T, v_tok.size(-2), v_tok.size(-1))       # [B*T, Lv-1, D]
        v_list, v_attn = self.slot_v(self.slots, v_tok_flat)                  # v_attn: [B*T, Nv, S]
        v_local = v_list[-1].reshape(B, T, -1, v_list[-1].size(-1))           # [B, T, S, D]
        v_attn  = v_attn.reshape(B, T, v_attn.size(1), v_attn.size(2))        # [B, T, Nv, S]

        loss_part = None
        loss_recon = None

        if use_metric:
            # v_local : [B, T, S, D]  →  帧均值  →  [B, S, D]
            v_mean = v_local.mean(dim=1)

            # 只有一句文本：直接一次 NCE
            t_p  = t_local[:, 0]              # [B, S, D]
            pw_p = part_w[:, 0] if part_w is not None else None  # [B, S] or None
            loss_part, _ = plot_obj.compute_nce(
                v_mean, t_p, pid, logit_scale * 5,
                local_flag=True, part_w=pw_p
            )

        # ---- Reconstruction with MlpDecoder (PLOT) ----
        loss_recon = None
        if use_recon:
            # 文本重建（与帧无关）：按句 batch 解码
            t_slots = t_local.reshape(B * P, self.num_slots, self.dim)                # [B*P,S,D]
            recon_t = self.recon_dec_t(t_slots)                                       # [B*P, n_txt, D]
            target_t = t_tok.reshape(B * P, n_txt, self.dim).detach()
            t_mse = F.mse_loss(recon_t, target_t, reduction='sum') / (B * P)          # 每句一次

            # 视频重建：按帧 batch 解码
            v_slots = v_local.reshape(B * T, self.num_slots, self.dim)                # [B*T,S,D]
            recon_v = self.recon_dec_v(v_slots)                                       # [B*T, n_vid, D]
            target_v = v_tok.reshape(B * T, n_vid, self.dim).detach()
            v_mse = F.mse_loss(recon_v, target_v, reduction='sum') / (B * T)          # 每帧一次

            loss_recon = recon_weight * (t_mse + v_mse)

        return {"loss_part": loss_part, "loss_recon": loss_recon}
