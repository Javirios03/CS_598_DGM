import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# from deepspeed.ops.adam import FusedAdam
from flow_matching.model import Flow
from dit.fused_add_dropout_scale import bias_dropout_add_scale_fused_train
from dit.rotary import Rotary, apply_rotary_pos_emb
from dit.transformer import DDiTBlock, DDitFinalLayer, LayerNorm, TimestepEmbedder

from einops import rearrange

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.encoder.num_heads

        self.norm1 = LayerNorm(config.encoder.hidden_dim)
        self.norm2 = LayerNorm(config.encoder.hidden_dim)

        # self.input_proj = nn.Linear(
        #     config.encoder.input_dim,
        #     config.encoder.hidden_dim
        # )

        self.attn_qkv = nn.Linear(
            config.encoder.hidden_dim,
            3 * config.encoder.hidden_dim, 
            bias=False
        )
        self.attn_out = nn.Linear(
            config.encoder.hidden_dim, 
            config.encoder.hidden_dim,
            bias=False
        )

        # self.ff = nn.Sequential(
        #     nn.Linear(config.encoder.hidden_dim, config.encoder.hidden_dim  * 4),
        #     nn.GELU(),
        #     nn.Linear(config.encoder.hidden_dim*4, config.encoder.hidden_dim),
        # )

    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]
        x = self.input_proj(x)
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x) # shape B x L X H
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        q, k, v = rearrange(qkv, 'b s three h d -> b h three s d', three=3, h=self.n_heads).unbind(2)
        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask[:,None,None,:] if attn_mask is not None else None
        )

        x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)
        x = self.attn_out(x) + x_skip

        x = self.norm2(x)
        ff_output = self.ff(x)
        
        x = x_skip + ff_output

        return torch.mean(x, dim=1)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_heads = config.encoder.num_heads

        self.norm1 = LayerNorm(config.encoder.hidden_dim)
        self.norm2 = LayerNorm(config.encoder.hidden_dim)

        self.input_proj = nn.Linear(
            config.encoder.input_dim,
            config.encoder.hidden_dim
        )

        self.attn_qkv = nn.Linear(
            config.encoder.hidden_dim,
            3 * config.encoder.hidden_dim, 
            bias=False
        )
        self.attn_out = nn.Linear(
            config.encoder.hidden_dim, 
            config.encoder.hidden_dim,
            bias=False
        )

        self.ff = nn.Sequential(
            nn.Linear(config.encoder.hidden_dim, config.encoder.hidden_dim  * 4),
            nn.GELU(),
            nn.Linear(config.encoder.hidden_dim*4, config.encoder.hidden_dim),
        )

    def forward(self, x, attn_mask=None):
        batch_size = x.shape[0]
        x = self.input_proj(x)
        x_skip = x
        x = self.norm1(x)

        qkv = self.attn_qkv(x) # shape B x L X H
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)

        q, k, v = rearrange(qkv, 'b s three h d -> b h three s d', three=3, h=self.n_heads).unbind(2)
        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=attn_mask[:,None,None,:] if attn_mask is not None else None
        )

        x = rearrange(x, 'b h s d -> b s (h d)', b=batch_size)
        x = self.attn_out(x) + x_skip

        x = self.norm2(x)
        ff_output = self.ff(x)
        
        x = x_skip + ff_output

        return torch.mean(x, dim=1)

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_proj = nn.Linear(
            config.encoder.input_dim,
            config.decoder.hidden_dim
        )
        self.emb_proj = nn.Linear(
            config.encoder.hidden_dim,
            config.decoder.hidden_dim
        )
        self.timestep_emb = TimestepEmbedder(config.decoder.conditioning_dim)
        self.rotary_emb = Rotary(config.decoder.hidden_dim * 2 // config.decoder.num_heads)
        self.blocks = nn.ModuleList([
            DDiTBlock(
                config.decoder.hidden_dim * 2,
                config.decoder.num_heads,
                config.decoder.conditioning_dim, 
                dropout=config.decoder.dropout
            ) for _ in range(config.decoder.depth)
        ])
        self.output_layer = DDitFinalLayer(
            config.decoder.hidden_dim * 2, 
            config.encoder.input_dim, 
            config.decoder.conditioning_dim
        )

    def forward(self, x, t, model_extras):
        set_emb = model_extras
        pt = self.input_proj(x)
        x  = pt
        set_emb = self.emb_proj(set_emb)
        
        x = torch.cat((pt, set_emb), dim=-1)
        c = F.silu(self.timestep_emb(t))
        rotary_cos_sin = self.rotary_emb(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x, rotary_cos_sin, c, attn_mask=None)
        x = self.output_layer(x, c)

        return x
    
class SetFlowModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.model = Flow(config, self.decoder)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

    def on_train_epoch_start(self):
        torch.cuda.empty_cache()

        self.model.train()
        
    def training_step(self, batch, batch_idx):
        x_1 = batch
        set_emb = self.encoder(x_1).unsqueeze(1).repeat(1, x_1.shape[1], 1)
        # set_emb = torch.rand_like(x_1)
        # print(set_emb.shape)
        # .repeat(1, x_1.shape[1])
        # print(set_emb.shape)
        # recon_points = x_1[:, torch.randint(0, x_1.size(1), (1,))]
        loss = self.model.get_loss(x_1, set_emb)
        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        return loss
    
    @torch.no_grad()
    def reconstruct(self, sample, batch_size, timesteps):
        set_emb = self.encoder(sample)
        # print(set_emb.shape)
        # set_emb = torch.rand_like(sample)
        # print(set_emb.shape)
        return self.model.sample(
            batch_size,
            set_emb.unsqueeze(1).repeat(1,sample.shape[1],1),
            timesteps=timesteps,
            step_size=0.001,
            device=sample.device
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.training.lr,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.beta1, self.config.training.beta2),
            # fused=True
        )

        return {
            "optimizer": optimizer,
        }