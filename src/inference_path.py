import argparse
import os
from pathlib import Path

import imageio
import numpy as np
import torch
from diffusers import CogVideoXPipeline
from natsort import natsorted

from utils import (CogVideoXAttnProcessor2, CogVideoXAttnProcessor2_0Mod,
                   CogVideoXBlock, CogVideoXBlockMod, apply_attn_processor,
                   apply_mod_block, export_to_video_imageio,
                   inversion_forward_process, inversion_reverse_process,
                   latent2tensor, load_video)


@torch.no_grad()
def main(args):
    model_path = args.pretrained_model_name_or_path
    dtype = torch.float16
    device = torch.device("cuda")
    num_inference_steps = 100
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to(device)

    if args.enable_tiling:
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    # extract videos from path
    video_paths = natsorted(list(Path(args.input_path).rglob("*.mp4")))
    assert len(video_paths) != 0, f"No videos found in {args.input_path}"

    for video_path in video_paths:
        frames_tensor = load_video(str(video_path), device=device, dtype=dtype)
        pipe.scheduler.set_timesteps(num_inference_steps)

        encoded_frames = (
            pipe.vae.encode(frames_tensor)[0].sample() * pipe.vae.config.scaling_factor
        )

        # Make sure that the attention processor is default for inversion
        apply_mod_block(
            pipe.transformer,
            CogVideoXBlock,
            CogVideoXAttnProcessor2,
            torch.arange(0, 30),
        )

        prompt_src = ""
        cfg_scale_src = 3.5
        skip = args.skip_value
        print("Inversion...")
        wt, zs, wts = inversion_forward_process(
            pipe,
            encoded_frames.permute(0, 2, 1, 3, 4),
            prompt=prompt_src,
            guidance_scale=cfg_scale_src,
            prog_bar=True,
            num_inference_steps=num_inference_steps,
            skip=0,
        )

        # Change attn processor for some blocks
        blocks_to_change = np.array(
            list(range(4)) + list(range(15, 30))
        )  # KV swap at beginning (0-4) and end (15-30)
        # blocks_to_change = np.array(list(range(4)))  # KV swap at the beginning )(0-4)
        # blocks_to_change = np.array([]])  # No KV swap
        apply_mod_block(
            pipe.transformer,
            CogVideoXBlockMod,
            CogVideoXAttnProcessor2_0Mod,
            blocks_to_change,
        )

        negative_prompt = args.negative_prompt
        cfg_scale_tar = args.editing_guidance_scale
        print("Deraining...")
        w0, _ = inversion_reverse_process(
            pipe,
            xT=wts[num_inference_steps - skip],
            prompt=prompt_src,
            guidance_scale=cfg_scale_tar,
            prog_bar=True,
            zs=zs[: (num_inference_steps - skip)],
            negative_prompt=negative_prompt,
        )
        ts = latent2tensor(w0, pipe)

        input_video_name = video_path.stem
        output_folder = f"outputs/{Path(args.input_path).name}/skip{str(skip)}_cfg{str(args.editing_guidance_scale)}_kv4_15"
        output_path = f"{output_folder}/{input_video_name}.mp4"
        os.makedirs(output_folder, exist_ok=True)
        export_to_video_imageio(
            pipe.video_processor.postprocess_video(ts, output_type="pil")[0],
            output_path,
            fps=8,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Model information
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="THUDM/CogVideoX-2b",
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        required=True,
        help="Path to input video",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        required=True,
        help="Negative prompt to be used",
    )
    parser.add_argument(
        "--skip_value",
        type=int,
        default=40,
        required=False,
        help="The number of steps to skip before starting editing.",
    )
    parser.add_argument(
        "--editing_guidance_scale",
        type=float,
        default=15,
        required=False,
        help="Guidance scale for applying negative guidance.",
    )
    parser.add_argument(
        "--enable_tiling",
        action="store_true",
        default=False,
        help="Whether to use VAE tiling.",
    )
    args = parser.parse_args()
    main(args)
