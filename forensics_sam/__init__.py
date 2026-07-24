import torch
import torch.nn as nn
import math

class LoRA_QKV(nn.Module):
    def __init__(self, ori_qkv, dim, r):
        super().__init__()
        self.activate = True
        self.ori_qkv = ori_qkv
        self.dim = dim
        self.a_q = nn.Linear(dim, r, bias=False)
        self.b_q = nn.Linear(r, dim, bias=False)
        self.a_k = nn.Linear(dim, r, bias=False)
        self.b_k = nn.Linear(r, dim, bias=False)
        self.a_v = nn.Linear(dim, r, bias=False)
        self.b_v = nn.Linear(r, dim, bias=False)
        nn.init.kaiming_uniform_(self.a_q.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_q.weight)
        nn.init.kaiming_uniform_(self.a_k.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_k.weight)
        nn.init.kaiming_uniform_(self.a_v.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b_v.weight)

    def set_shared_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        qkv = self.ori_qkv(x)  # [B,H,W,3C]
        if self.activate:
            qkv[:, :, :, : self.dim] +=self.b_q(self.a_q(x))
            qkv[:, :, :, self.dim:-self.dim] +=self.b_k(self.a_k(x))
            qkv[:, :, :, -self.dim:] += self.b_v(self.a_v(x))
        return qkv

class LoRA_FFN(nn.Module):
    def __init__(self, ori_lin, in_features, out_features, r):
        super().__init__()
        self.activate = True
        self.ori_lin = ori_lin
        self.a = nn.Linear(in_features, r, bias=False)
        self.b = nn.Linear(r, out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)

    def set_shared_activate(self, activate):
        self.activate = activate

    def forward(self, x):
        out = self.ori_lin(x)
        if self.activate:
            out = out + self.b(self.a(x))
        return out


class ForensicsSAM(nn.Module):
    def __init__(self, sam_model, r=8, lora_layer=None):
        super().__init__()
        assert r > 0
        self.lora_layer = lora_layer or list(range(len(sam_model.image_encoder.blocks)))

        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        blk_num = len(sam_model.image_encoder.blocks)
        if blk_num == 32:
            self.global_attn_index = [7, 15, 23, 31]
        elif blk_num == 24:
            self.global_attn_index = [5, 11, 17, 23]
        elif blk_num == 12:
            self.global_attn_index = [2, 5, 8, 11]

        ''' Shared Forgery Experts '''
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            dim = w_qkv_linear.in_features
            blk.attn.qkv = LoRA_QKV(w_qkv_linear, dim, r)

            w_lin1 = blk.mlp.lin1
            in_features = w_lin1.in_features
            out_features = w_lin1.out_features
            blk.mlp.lin1 = LoRA_FFN(w_lin1, in_features, out_features, r)

        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False

        self.sam = sam_model

    def forward(self, images):

        image_embeddings = self.sam.image_encoder(images)
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None,
            boxes=None,
            masks=None,
        )
        device = image_embeddings.device
        mask_prediction, iou_prediction = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe().to(device),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )
        return mask_prediction