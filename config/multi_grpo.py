import os
import runpy
from pathlib import Path


def get_config(task):
    config = runpy.run_path(str(Path(__file__).with_name("base.py")))["get_config"]()
    if task == "ocr_color":
        config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
        config.dataset = "dataset/OCR-Color-10"
        config.prompt_fn = "ocr_with_color"
        config.sample.num_steps = 10
        config.sample.eval_num_steps = 40
        config.sample.guidance_scale = 4.5
        config.sample.num_batches_per_epoch = 16
        config.train.beta = 0.04
        config.reward_wise_grouping = True
        config.reward_fn = {"ocr_with_color": 1.0, "vqa_color": 1.0, "pickscore": 1.0}
    elif task == "pickscore":
        config.pretrained.model = "black-forest-labs/FLUX.1-dev"
        config.dataset = "dataset/PickScore-25k"
        config.prompt_fn = "general_ocr"
        config.sample.num_steps = 6
        config.sample.eval_num_steps = 28
        config.sample.guidance_scale = 3.5
        config.sample.num_batches_per_epoch = 48
        config.reward_fn = {"pickscore": 1.0}
    else:
        raise ValueError(f"Unknown Multi-GRPO task: {task}")
    config.save_freq = config.eval_freq = 60 if task == "ocr_color" else 30
    # (one-based denoising update, children per active path).
    config.branches = ((2, 3), (4, 2))
    config.resolution = 512
    config.mixed_precision = "bf16"
    config.sample.noise_level = 0.7
    config.sample.num_image_per_prompt = 8
    config.sample.train_batch_size = config.train.batch_size = 1
    config.sample.test_batch_size = 16
    config.sample.same_latent = True
    config.train.ema = True
    config.train.timestep_fraction = 0.99
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.per_prompt_stat_tracking = False
    config.save_dir = os.path.join("logs", task)
    return config
