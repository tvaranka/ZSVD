from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple

import imageio
import numpy as np
import PIL
import torch
from diffusers.utils.torch_utils import maybe_allow_in_graph
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm


@torch.no_grad()
def load_video(
    video_path: str,
    num_frames: int = 49,
    height: int = 480,
    width: int = 720,
    device=None,
    dtype=None,
) -> torch.Tensor:
    if video_path.endswith(".mp4"):
        video_reader = imageio.get_reader(video_path, "ffmpeg")
    else:
        video_reader = [
            Image.open(path) for path in sorted(Path(video_path).glob("*.jpg"))
        ]

    frames = [transforms.ToTensor()(frame) for frame in video_reader]
    video_reader.close() if video_path.endswith(".mp4") else None

    frames_tensor = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)
    frames_tensor = frames_tensor * 2 - 1
    # only take num_fames
    frames_tensor = frames_tensor[:, :, :num_frames]
    frames_tensor = F.interpolate(frames_tensor, (num_frames, height, width)).to(
        device, dtype
    )
    return frames_tensor


def tensor2img(x0: torch.Tensor, index=None):
    x0 = ((x0[0][index] + 1) / 2).clip_(0, 1)
    return Image.fromarray(
        (np.array(x0.permute(1, 2, 0).cpu().float()) * 255).astype("uint8")
    )


@torch.inference_mode()
def latent2tensor(latent: torch.Tensor, pipe) -> torch.Tensor:
    latent = latent.permute(
        0, 2, 1, 3, 4
    )  # [batch_size, num_channels, num_frames, height, width]
    latent = 1 / pipe.vae.config.scaling_factor * latent
    frames = pipe.vae.decode(latent).sample.cpu()
    return frames


def tensor2uint(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.cpu()[0].permute(1, 2, 3, 0).float()
    tensor = ((tensor + 1) / 2) * 255.0
    return tensor


def psnr(gt: torch.Tensor, prediction: torch.Tensor) -> float:
    gt = tensor2uint(gt)
    prediction = tensor2uint(prediction)
    mse = torch.mean((gt - prediction) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def psnr_frames(gt: torch.Tensor, prediction: torch.Tensor) -> float:
    gt = tensor2uint(gt)
    prediction = tensor2uint(prediction)
    psnrs = []
    for i in range(len(gt)):
        mse = torch.mean((gt[i] - prediction[i]) ** 2)
        if mse == 0:
            return 100
        max_pixel = 255.0
        psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
        psnrs.append(psnr)
    return torch.tensor(psnrs)


@torch.no_grad()
def inversion_forward_process(
    pipe,
    x0,
    prog_bar=False,
    prompt="",
    guidance_scale=3.5,
    num_inference_steps=100,
    negative_prompt="",
    skip=0,
):
    device = pipe.device
    dtype = pipe.dtype
    timesteps = pipe.scheduler.timesteps.to(device)
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        negative_prompt,
        True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=pipe.tokenizer.model_max_length,
        device=device,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    variance_noise_shape = [num_inference_steps] + list(x0.shape[1:])
    xts = torch.cat(
        [x0]
        + [
            pipe.scheduler.add_noise(x0, torch.randn_like(x0), t)
            for t in reversed(timesteps)
        ]
    )
    zts = torch.zeros(size=variance_noise_shape, device=device, dtype=dtype)
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = x0.to(pipe.transformer.dtype)
    timesteps = timesteps[: (num_inference_steps - skip)]
    op = tqdm(timesteps) if prog_bar else timesteps

    for t in op:
        idx = num_inference_steps - t_to_idx[int(t)] - 1
        xt = xts[idx + 1][None]

        timestep = t.expand(xt.shape[0])

        latent_model_input = torch.cat([xt] * 2)
        model_output = pipe.transformer(
            latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds
        )[0]
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        xtm1 = xts[idx][None]
        prev_timestep = (
            t
            - pipe.scheduler.config.num_train_timesteps
            // pipe.scheduler.num_inference_steps
        )
        variance = pipe.scheduler._get_variance(t, prev_timestep)

        mu_xt = pipe.scheduler.step(noise_pred, t, xt).prev_sample

        z = (xtm1 - mu_xt) / (variance**0.5)
        zts[idx] = z
        xts[idx] = mu_xt + (variance**0.5) * z

    zts[0] = torch.zeros_like(zts[0])

    return xt, zts, xts


@torch.no_grad()
def inversion_reverse_process(
    pipe,
    xT,
    prompt="",
    guidance_scale=None,
    prog_bar=False,
    zs=None,
    negative_prompt="",
    negative_prompt_embeddings=None,
):

    device = pipe.device
    dtype = pipe.dtype
    timesteps = pipe.scheduler.timesteps.to(device)
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        negative_prompt,
        True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=pipe.tokenizer.model_max_length,
        device=device,
    )
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    timesteps = timesteps[-zs.shape[0] :]
    op = tqdm(timesteps) if prog_bar else timesteps
    t_to_idx = {int(v): k for k, v in enumerate(timesteps)}
    xt = xT[None]

    for t in op:
        idx = (
            pipe.scheduler.num_inference_steps
            - t_to_idx[int(t)]
            - (pipe.scheduler.num_inference_steps - zs.shape[0] + 1)
        )
        t = t.expand(xt.shape[0])

        latent_model_input = torch.cat([xt] * len(prompt_embeds))

        model_output = pipe.transformer(
            latent_model_input, timestep=t, encoder_hidden_states=prompt_embeds
        )[0]
        noise_pred_uncond, noise_pred_text = model_output.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        z = zs[idx]
        xt = pipe.scheduler.step(noise_pred, t, xt).prev_sample
        prev_timestep = (
            t
            - pipe.scheduler.config.num_train_timesteps
            // pipe.scheduler.num_inference_steps
        )
        variance = pipe.scheduler._get_variance(t, prev_timestep)
        sigma_z = variance ** (0.5) * z
        xt = xt + sigma_z
        xt = xt.to(device, dtype)
    return xt, zs


def export_to_video_imageio(
    video_frames: List[np.ndarray] | List[PIL.Image.Image],
    output_video_path: str = None,
    fps: int = 8,
) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name
    if isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for frame in video_frames:
            writer.append_data(frame)
    return output_video_path


class CogVideoXAttnProcessor2_0Mod:

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        # Add cross-condition hidden feature, eq. (6)
        new_hs = torch.cat(
            [hidden_states[1, :text_seq_length], hidden_states[0, text_seq_length:]]
        )
        hidden_states = torch.cat([hidden_states, new_hs[None]])

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from .embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        # Extract only kv from text part
        key[0, :text_seq_length] = key[2, :text_seq_length].clone()
        value[0, :text_seq_length] = value[2, :text_seq_length].clone()

        # back to batch size 2
        query = query[:2]
        key = key[:2]
        value = value[:2]
        batch_size = 2

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


class CogVideoXAttnProcessor2:

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "CogVideoXAttnProcessor requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            from .embeddings import apply_rotary_emb

            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class CogVideoXBlockMod(nn.Module):
    def __init__(
        self,
        norm1,
        attn1,
        norm2,
        ff,
    ):
        super().__init__()
        self.norm1 = norm1
        self.attn1 = attn1
        self.norm2 = norm2
        self.ff = ff

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs=None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = (
            self.norm1(hidden_states, encoder_hidden_states, temb)
        )

        # attention
        _, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        # #MODIFIED EDITING
        # #attention 2
        attn_hidden_states, _ = self.attn2(
            hidden_states=norm_hidden_states.clone(),
            encoder_hidden_states=norm_encoder_hidden_states.clone(),
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        )

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = (
            self.norm2(hidden_states, encoder_hidden_states, temb)
        )

        # feed-forward
        norm_hidden_states = torch.cat(
            [norm_encoder_hidden_states, norm_hidden_states], dim=1
        )
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        )

        return hidden_states, encoder_hidden_states


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    def __init__(
        self,
        norm1,
        attn1,
        norm2,
        ff,
    ):
        super().__init__()
        self.norm1 = norm1
        self.attn1 = attn1
        self.norm2 = norm2
        self.ff = ff

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_kwargs=None,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = (
            self.norm1(hidden_states, encoder_hidden_states, temb)
        )

        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states
        )

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = (
            self.norm2(hidden_states, encoder_hidden_states, temb)
        )

        # feed-forward
        norm_hidden_states = torch.cat(
            [norm_encoder_hidden_states, norm_hidden_states], dim=1
        )
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = (
            encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]
        )

        return hidden_states, encoder_hidden_states


def apply_mod_block(transformer, used_block, attn_processor, blocks: List[int]):
    for i in blocks:
        block = transformer.transformer_blocks[i]
        transformer.transformer_blocks[i] = used_block(
            block.norm1, block.attn1, block.norm2, block.ff
        )
        transformer.transformer_blocks[i].attn2 = deepcopy(
            transformer.transformer_blocks[i].attn1
        )
        transformer.transformer_blocks[i].attn2.processor = attn_processor()


def apply_attn_processor(transformer, attn_processor, blocks: List[int]):
    for i in blocks:
        transformer.transformer_blocks[i].attn1.processor = attn_processor()
