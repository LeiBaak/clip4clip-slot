from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
from torch import nn

from modules.until_module import PreTrainedModel, AllGather, CrossEn
from modules.module_cross import CrossModel, CrossConfig, Transformer as TransformerClip

from modules.module_clip import CLIP, convert_weights
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from modules.slot_adapter import SlotAdapter

logger = logging.getLogger(__name__)
allgather = AllGather.apply

class CLIP4ClipPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, cross_config, *inputs, **kwargs):
        super(CLIP4ClipPreTrainedModel, self).__init__(cross_config)
        self.cross_config = cross_config
        self.clip = None
        self.cross = None

    @classmethod
    def from_pretrained(cls, cross_model_name, state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        if state_dict is None: state_dict = {}
        pretrained_clip_name = "ViT-B/32"
        if hasattr(task_config, 'pretrained_clip_name'):
            pretrained_clip_name = task_config.pretrained_clip_name
        clip_state_dict = CLIP.get_config(pretrained_clip_name=pretrained_clip_name)
        for key, val in clip_state_dict.items():
            new_key = "clip." + key
            if new_key not in state_dict:
                state_dict[new_key] = val.clone()

        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(cross_config, clip_state_dict, *inputs, **kwargs)

        use_psa = hasattr(task_config, 'use_psa') and task_config.use_psa
        model.use_psa = bool(use_psa)

        # 与 PLOT 配套的默认值（和源码一致）
        num_slots = getattr(task_config, 'num_slots', 8)
        num_iters = getattr(task_config, 'num_slot_iters', 5)
        model.start_recon  = getattr(task_config, 'start_recon', 0)      # PLOT: default 5
        model.start_metric = getattr(task_config, 'start_metric', 10)    # PLOT: default 15
        model.recon_weight   = getattr(task_config, 'recon_weight', 0.01)  # PLOT 中固定 0.01
        model.partnce_weight = getattr(task_config, 'partnce_weight', 1.0) # PLOT 没单独权重，等价 1.0

        if model.use_psa:
            dim = clip_state_dict['text_projection'].size(1)
            model.plot = SlotAdapter(dim=dim, num_slots=num_slots, num_iters=num_iters)
        else:
            model.plot = None

        plot_temperature = getattr(task_config, 'plot_temperature', 0.015)   # 可配
        plot_logit_scale = torch.tensor(1.0 / plot_temperature, dtype=torch.float32)
        # 用 buffer 保存，能随 .to(device) 与 checkpoint 同步；且不会被 optimizer 更新
        model.register_buffer('plot_logit_scale', plot_logit_scale)

        ## ===> Initialization trick [HARD CODE]
        if model.linear_patch == "3d":
            contain_conv2 = False
            for key in state_dict.keys():
                if key.find("visual.conv2.weight") > -1:
                    contain_conv2 = True
                    break
            if contain_conv2 is False and hasattr(model.clip.visual, "conv2"):
                cp_weight = state_dict["clip.visual.conv1.weight"].clone()
                kernel_size = model.clip.visual.conv2.weight.size(2)
                conv2_size = model.clip.visual.conv2.weight.size()
                conv2_size = list(conv2_size)

                left_conv2_size = conv2_size.copy()
                right_conv2_size = conv2_size.copy()
                left_conv2_size[2] = (kernel_size - 1) // 2
                right_conv2_size[2] = kernel_size - 1 - left_conv2_size[2]

                left_zeros, right_zeros = None, None
                if left_conv2_size[2] > 0:
                    left_zeros = torch.zeros(*tuple(left_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)
                if right_conv2_size[2] > 0:
                    right_zeros = torch.zeros(*tuple(right_conv2_size), dtype=cp_weight.dtype, device=cp_weight.device)

                cat_list = []
                if left_zeros != None: cat_list.append(left_zeros)
                cat_list.append(cp_weight.unsqueeze(2))
                if right_zeros != None: cat_list.append(right_zeros)
                cp_weight = torch.cat(cat_list, dim=2)

                state_dict["clip.visual.conv2.weight"] = cp_weight

        if model.sim_header == 'tightTransf':
            contain_cross = False
            for key in state_dict.keys():
                if key.find("cross.transformer") > -1:
                    contain_cross = True
                    break
            if contain_cross is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["cross.embeddings.position_embeddings.weight"] = val.clone()
                        continue
                    if key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])

                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict["cross."+key] = val.clone()
                            continue

        if model.sim_header == "seqLSTM" or model.sim_header == "seqTransf":
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in clip_state_dict.items():
                    if key == "positional_embedding":
                        state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    if model.sim_header == "seqTransf" and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        # cut from beginning
                        if num_layer < task_config.cross_num_hidden_layers:
                            state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        ## <=== End of initialization trick

        if state_dict is not None:
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model

def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)

def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config

def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]

class CLIP4Clip(CLIP4ClipPreTrainedModel):
    def __init__(self, cross_config, clip_state_dict, task_config):
        super(CLIP4Clip, self).__init__(cross_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.loose_type = False
        if self._stage_one and check_attr('loose_type', self.task_config):
            self.loose_type = True
            show_log(task_config, "Test retrieval by loose type.")

        # CLIP Encoders: From OpenAI: CLIP [https://github.com/openai/CLIP] ===>
        vit = "visual.proj" in clip_state_dict
        assert vit
        if vit:
            vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len(
                [k for k in clip_state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"visual.layer{b}"))) for b in
                            [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = clip_state_dict["text_projection"].shape[1]
        context_length = clip_state_dict["positional_embedding"].shape[0]
        vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
        transformer_width = clip_state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in clip_state_dict if k.startswith(f"transformer.resblocks")))

        show_log(task_config, "\t embed_dim: {}".format(embed_dim))
        show_log(task_config, "\t image_resolution: {}".format(image_resolution))
        show_log(task_config, "\t vision_layers: {}".format(vision_layers))
        show_log(task_config, "\t vision_width: {}".format(vision_width))
        show_log(task_config, "\t vision_patch_size: {}".format(vision_patch_size))
        show_log(task_config, "\t context_length: {}".format(context_length))
        show_log(task_config, "\t vocab_size: {}".format(vocab_size))
        show_log(task_config, "\t transformer_width: {}".format(transformer_width))
        show_log(task_config, "\t transformer_heads: {}".format(transformer_heads))
        show_log(task_config, "\t transformer_layers: {}".format(transformer_layers))

        self.linear_patch = '2d'
        if hasattr(task_config, "linear_patch"):
            self.linear_patch = task_config.linear_patch
            show_log(task_config, "\t\t linear_patch: {}".format(self.linear_patch))

        # use .float() to avoid overflow/underflow from fp16 weight. https://github.com/openai/CLIP/issues/40
        cut_top_layer = 0
        show_log(task_config, "\t cut_top_layer: {}".format(cut_top_layer))
        self.clip = CLIP(
            embed_dim,
            image_resolution, vision_layers-cut_top_layer, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers-cut_top_layer,
            linear_patch=self.linear_patch
        ).float()

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in clip_state_dict:
                del clip_state_dict[key]

        convert_weights(self.clip)
        # <=== End of CLIP Encoders

        self.sim_header = 'meanP'
        if hasattr(task_config, "sim_header"):
            self.sim_header = task_config.sim_header
            show_log(task_config, "\t sim_header: {}".format(self.sim_header))
        if self.sim_header == "tightTransf": assert self.loose_type is False

        cross_config.max_position_embeddings = context_length
        if self.loose_type is False:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers", self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder
            self.similarity_dense = nn.Linear(cross_config.hidden_size, 1)

        if self.sim_header == "seqLSTM" or self.sim_header == "seqTransf":
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings, cross_config.hidden_size)
        if self.sim_header == "seqTransf":
            self.transformerClip = TransformerClip(width=transformer_width, layers=self.task_config.cross_num_hidden_layers,
                                                   heads=transformer_heads, )
        if self.sim_header == "seqLSTM":
            self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size, hidden_size=cross_config.hidden_size,
                                       batch_first=True, bidirectional=False, num_layers=1)

        self.loss_fct = CrossEn()

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None):
        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])

        # T x 3 x H x W
        video = torch.as_tensor(video).float()
        b, pair, bs, ts, channel, h, w = video.shape
        video = video.view(b * pair * bs * ts, channel, h, w)
        video_frame = bs * ts

        sequence_output, visual_output, extras = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True, video_frame=video_frame)

        if self.training:
            loss = 0.
            sim_matrix, *_tmp = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                    shaped=True, loose_type=self.loose_type)
            sim_loss1 = self.loss_fct(sim_matrix)
            sim_loss2 = self.loss_fct(sim_matrix.T)
            sim_loss = (sim_loss1 + sim_loss2) / 2
            loss += sim_loss
            log = {"loss_global": float(sim_loss.detach().cpu())}

            if (self.plot is not None) and (extras is not None):
                bs_pair = input_ids.size(0)

                # 来自一次 encode_* 的 token/hidden（包含 CLS）；Bridge 内部会自行去 CLS
                txt_tokens_all = extras["txt_hidden"]                  # [B, P, Lt, D]
                vid_tokens_all = extras["vid_hidden"]                  # [B, T, Lv, D]

                # 文本 mask（包含 CLS）；Bridge 内部会去 CLS
                txt_mask_all   = attention_mask.view(bs_pair, -1, attention_mask.size(-1))   # [B, P, Lt]

                # 与 PLOT 同步的启用时机
                cur_epoch  = getattr(self, "current_epoch", 0)
                use_metric = (cur_epoch > getattr(self, "start_metric", 9))   # 默认 >15
                use_recon  = (cur_epoch > getattr(self, "start_recon", 0))    # 默认 >5

                # pid：若没有 ReID ID，就用 arange(B) 让每个 (b,·) 仅与自身成正
                pid = getattr(self, "pids", None)
                if (pid is None) or (pid.numel() != bs_pair):
                    pid = torch.arange(bs_pair, device=sequence_output.device)

                outs = self.plot(
                    vid_tokens_all=vid_tokens_all,
                    txt_tokens_all=txt_tokens_all,
                    txt_mask_all=txt_mask_all,
                    txt_global_all=sequence_output,
                    logit_scale=self.plot_logit_scale,
                    pid=pid,
                    use_metric=use_metric,
                    use_recon=use_recon,
                    recon_weight=getattr(self, "recon_weight", 0.01),
                )

                loss_part  = outs.get("loss_part", None)
                loss_recon = outs.get("loss_recon", None)

                if (loss_part is not None) and use_metric:
                    loss = loss + self.partnce_weight * loss_part
                    log["loss_part_raw"] = float(loss_part.detach().cpu())
                    log["loss_part_w"]   = float((self.partnce_weight * loss_part).detach().cpu())
                else:
                    log["loss_part_raw"] = 0.0
                    log["loss_part_w"]   = 0.0

                if (loss_recon is not None) and use_recon:
                    loss = loss + self.recon_weight * loss_recon
                    log["loss_recon_raw"] = float(loss_recon.detach().cpu())
                    log["loss_recon_w"]   = float((self.recon_weight * loss_recon).detach().cpu())
                else:
                    log["loss_recon_raw"] = 0.0
                    log["loss_recon_w"]   = 0.0

                # 3) 占比（按“加权后”的贡献来算）
                tot = float(loss.detach().cpu())
                pg = log["loss_global"] / tot if tot > 0 else 0.0
                pp = log["loss_part_w"] / tot if tot > 0 else 0.0
                pr = log["loss_recon_w"] / tot if tot > 0 else 0.0

                log.update({
                    "loss_total": tot,
                    "prop_global": pg,             # 全局对比（加权后 = 本身）
                    "prop_part":   pp,             # 部件对比（乘权重后）
                    "prop_recon":  pr,             # 重建（乘权重后）
                })
                self._last_log = log

            return loss
        else:
            return None

    # ========== 评估用：文本局部槽特征 ==========
    @torch.no_grad()
    def encode_text_local(self, input_ids, attention_mask):
        """
        返回：
            txt_global: [N_txt, D]
            t_local:   [N_txt, S, D]
            part_w:    [N_txt, S] or None
        """
        self.eval()

        # --- 关键：严格对齐 eval 流水线，支持 [B,S,L] / [N,L] ---
        if input_ids.dim() == 3:
            b, s, l = input_ids.shape
            input_ids = input_ids.reshape(b * s, l)
            if attention_mask is not None and attention_mask.dim() == 3:
                attention_mask = attention_mask.reshape(b * s, l)
        elif input_ids.dim() != 2:
            raise ValueError(f"encode_text_local expects [N,L] or [B,S,L], got {tuple(input_ids.shape)}")

        # 1) 先取 pooled 和所有 token hidden（与原 get_sequence_output 的 pooled 一致）
        txt_global, txt_hidden = self.clip.encode_text(input_ids, return_hidden=True)  # [N,D], [N,L,D]
        txt_global = txt_global.float()
        txt_hidden = txt_hidden.float()

        # 2) 去 CLS，做文本侧 SlotAttention（一次性 batch）
        t_tok = txt_hidden[:, 1:, :]
        t_msk = attention_mask[:, 1:] if attention_mask is not None else None

        assert self.plot is not None, "PLOT bridge not attached"
        t_list, _ = self.plot.slot_t(self.plot.slots, t_tok, mask=t_msk)  # list[it][N,S,D]
        t_local = t_list[-1]                                              # [N,S,D]

        # 3) 可选：文本槽权重
        if hasattr(self.plot, "text_slot_classifier"):
            part_w = torch.softmax(self.plot.text_slot_classifier(txt_global), dim=1)  # [N,S]
        else:
            part_w = None

        return txt_global, t_local, part_w


    # ========== 评估用：视频局部槽特征 ==========
    @torch.no_grad()
    def encode_video_local(self, video, video_frame=None):
        """
        输入:
            video: [..., 3, H, W]，常见 7D: [B, P, BS, TS, 3, H, W]
            video_frame: 每个样本的帧数（如 12）。如果 None，则尝试从形状推断。
        返回:
            vid_global: [Nv, D]   —— 视频全局向量（按帧平均后，每个视频/样本一条）
            v_local:   [Nv, S, D] —— 槽级局部表示（对时空 token 聚合后的 S 个槽）
        """
        self.eval()

        # 基本检查
        assert video.dim() >= 4, f"expect [...,3,H,W], got {tuple(video.shape)}"
        *prefix, C, H, W = video.shape
        assert C == 3, f"expect channel=3, got {C}"
        assert video_frame is not None and int(video_frame) > 0, "video_frame must be provided and > 0"

        # 展平到 [N,3,H,W]，N = 所有前缀维度连乘
        N_frames = 1
        for d in prefix:
            N_frames *= int(d)
        vf = int(video_frame)
        assert N_frames % vf == 0, f"frames {N_frames} not divisible by video_frame {vf}"
        Nv = N_frames // vf

        # 2) 展平成 4D，送入 CLIP（完全对齐旧路径）
        video_4d = video.contiguous().view(N_frames, C, H, W)   # [N,3,H,W]
        vid_global, vid_hidden = self.clip.encode_image(
            video_4d, return_hidden=True, video_frame=vf
        )  # vid_global: [N, D]；vid_hidden: [N, L, D]（含 CLS）
        vid_global = vid_global.float()
        vid_hidden = vid_hidden.float()

        # 3) 回拼成“每个样本 Nv 一条”的全局特征（按帧平均）
        D = vid_global.shape[-1]
        vid_global = vid_global.reshape(Nv, vf, D).mean(dim=1)     # [Nv, D]

        # 4) 取去 CLS 的时空 token，并把“帧 × patch”展平到一个维度，供 slot_v 聚合
        #    旧路径里相当于把每帧的空间 token 拼起来，然后再做局部聚合/匹配
        L = vid_hidden.shape[1]                                  # 含 CLS 的长度
        assert L >= 2, f"unexpected visual seq length L={L}"
        v_tok = vid_hidden[:, 1:, :]                            # [N, L-1, D] 去掉 CLS
        v_tok = v_tok.reshape(Nv, vf * (L - 1), D)                 # [Nv, T*Sp, D] 时空展平

        # 5) 槽聚合（保持与文本侧的 slot 使用一致）
        assert self.plot is not None, "PLOT bridge not attached"
        v_list, _ = self.plot.slot_v(self.plot.slots, v_tok, mask=None)  # list[it][Nv,S,D]
        v_local = v_list[-1]                                             # [Nv, S, D]

        return vid_global, v_local


    # ========== 评估用：全局+槽级相似度融合 ==========
    @torch.no_grad()
    def fuse_similarity_with_slots(self, txt_global, t_local, part_w, vid_global, v_local,
                                use_part_w=True, start_metric=None):
        """
        输入：
        txt_global: [N_txt,D]   t_local: [N_txt,S,D]   part_w: [N_txt,S] or None
        vid_global: [N_vid,D]   v_local: [N_vid,S,D]
        输出：
        similarity: [N_txt, N_vid]  （全局 + 局部）
        """
        self.eval()
        import torch.nn.functional as F

        # 全局相似度（L2 归一化后内积）
        q = F.normalize(txt_global, dim=1)
        g = F.normalize(vid_global, dim=1)
        sim_global = q @ g.t()                                                  # [N_txt,N_vid]

        # 槽级相似度（逐槽对齐，L2 归一化）
        t_norm = F.normalize(t_local, dim=2)                                    # [N_txt,S,D]
        v_norm = F.normalize(v_local, dim=2)                                    # [N_vid,S,D]
        # (b,s,d) · (n,s,d) -> (b,s,n)
        part_sim = torch.einsum('bsd,nsd->bsn', t_norm, v_norm)                 # [N_txt,S,N_vid]

        # 融合方式：有 part_w 就加权，否则按槽 sigmoid 累加（与 PLOT eval 一致）
        if use_part_w and (part_w is not None):
            sim_part = torch.einsum('bsn,bs->bn', part_sim, part_w)             # [N_txt,N_vid]
        else:
            sim_part = torch.sigmoid(part_sim[:, 0])
            for s in range(1, part_sim.size(1)):
                sim_part = sim_part + torch.sigmoid(part_sim[:, s])             # [N_txt,N_vid]

        # 与 PLOT 一致的门控：默认直接加；若传入 start_metric，则“开启槽”的条件为 True 才加
        if start_metric is None:
            similarity = sim_global + sim_part
        else:
            similarity = sim_global + sim_part  # 评估阶段通常直接融合；若严格复刻，可外部判断 epoch>start_metric
        return similarity

    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = input_ids.size(0)

        if getattr(self, 'use_psa', False):
            txt_global, txt_hidden = self.clip.encode_text(input_ids, return_hidden=True)         # [B*pair, D], [B*pair, Lt, D]
            txt_global = txt_global.float()
            txt_hidden = txt_hidden.float()
            vid_global, vid_hidden = self.clip.encode_image(video, return_hidden=True, video_frame=video_frame)  # [B*frames, D], [B*frames, Lv, D]
            vid_global.float()
            vid_hidden = vid_hidden.float()
        else:
            txt_global = self.clip.encode_text(input_ids).float()                                         # [B*pair, D]
            vid_global = self.clip.encode_image(video, video_frame=video_frame).float()                   # [B*frames, D]
            txt_hidden = vid_hidden = None

        # 还原成 [bs_pair, pair, ...] / [bs_pair, frames, ...]
        sequence_output = txt_global.view(bs_pair, -1, txt_global.size(-1))       # 文本 pooled
        visual_output   = vid_global.view(bs_pair, -1, vid_global.size(-1))       # 视频 pooled（逐帧）

        extras = None
        if txt_hidden is not None and vid_hidden is not None:
            # 文本 hidden: [bs_pair, pair, Lt, D]
            txt_hidden = txt_hidden.view(bs_pair, -1, txt_hidden.size(1), txt_hidden.size(2))
            # 视频 hidden: [bs_pair, frames, Lv, D]
            vid_hidden = vid_hidden.view(bs_pair, -1, vid_hidden.size(1), vid_hidden.size(2))
            extras = {
                "txt_hidden": txt_hidden,
                "vid_hidden": vid_hidden,
            }

        return sequence_output, visual_output, extras

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):

        concat_features = torch.cat((sequence_output, visual_output), dim=1)  # concatnate tokens and frames
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        return cross_output, pooled_output, concat_mask

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)
        return text_out

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum
        return video_out

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        if sim_header == "meanP":
            # Default: Parameter-free type
            pass
        elif sim_header == "seqLSTM":
            # Sequential type: LSTM
            visual_output_original = visual_output
            visual_output = pack_padded_sequence(visual_output, torch.sum(video_mask, dim=-1).cpu(),
                                                 batch_first=True, enforce_sorted=False)
            visual_output, _ = self.lstm_visual(visual_output)
            if self.training: self.lstm_visual.flatten_parameters()
            visual_output, _ = pad_packed_sequence(visual_output, batch_first=True)
            visual_output = torch.cat((visual_output, visual_output_original[:, visual_output.size(1):, ...].contiguous()), dim=1)
            visual_output = visual_output + visual_output_original
        elif sim_header == "seqTransf":
            # Sequential type: Transformer Encoder
            visual_output_original = visual_output
            seq_length = visual_output.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=visual_output.device)
            position_ids = position_ids.unsqueeze(0).expand(visual_output.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            visual_output = visual_output + frame_position_embeddings

            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)
            visual_output = visual_output.permute(1, 0, 2)  # NLD -> LND
            visual_output = self.transformerClip(visual_output, extended_video_mask)
            visual_output = visual_output.permute(1, 0, 2)  # LND -> NLD
            visual_output = visual_output + visual_output_original

        if self.training:
            visual_output = allgather(visual_output, self.task_config)
            video_mask = allgather(video_mask, self.task_config)
            sequence_output = allgather(sequence_output, self.task_config)
            torch.distributed.barrier()

        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
        visual_output = self._mean_pooling_for_similarity_visual(visual_output, video_mask)
        visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

        sequence_output = sequence_output.squeeze(1)
        sequence_output = sequence_output / sequence_output.norm(dim=-1, keepdim=True)

        logit_scale = self.clip.logit_scale.exp()
        retrieve_logits = logit_scale * torch.matmul(sequence_output, visual_output.t())
        return retrieve_logits

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        sequence_output, visual_output = sequence_output.contiguous(), visual_output.contiguous()

        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []

        step_size = b_text      # set smaller to reduce memory cost
        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        # due to clip text branch retrun the last hidden
        attention_mask = torch.ones(sequence_output.size(0), 1)\
            .to(device=attention_mask.device, dtype=attention_mask.dtype)

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)

        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        contrastive_direction = ()
        if loose_type:
            assert self.sim_header in ["meanP", "seqLSTM", "seqTransf"]
            retrieve_logits = self._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header=self.sim_header)
        else:
            assert self.sim_header in ["tightTransf"]
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask, )

        return retrieve_logits, contrastive_direction
