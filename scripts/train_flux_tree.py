from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import FluxPipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.rewards
from flow_grpo.tree_advantages import local_tree_advantages
from flow_grpo.diffusers_patch.flux_tree_pipeline_with_logprob import tree_pipeline_with_logprob
from flow_grpo.diffusers_patch.flux_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/multi_grpo.py:pickscore", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization

        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()

            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])

            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs

def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
        text_ids = text_ids.to(device)
    return prompt_embeds, pooled_prompt_embeds

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

def compute_log_prob(transformer, pipeline, sample, j, config):
    packed_noisy_model_input = sample["latents"][j]
    device = packed_noisy_model_input.device
    dtype = packed_noisy_model_input.dtype
    if transformer.module.config.guidance_embeds:
        guidance = torch.tensor([config.sample.guidance_scale], device=device)
        guidance = guidance.expand(packed_noisy_model_input.shape[0])
    else:
        guidance = None

    model_pred = transformer(
        hidden_states=packed_noisy_model_input,
        timestep=sample["timesteps"][j:j+1] / 1000,
        guidance=guidance,
        pooled_projections=sample["pooled_prompt_embeds"].reshape(1, -1).expand(packed_noisy_model_input.shape[0], -1),
        encoder_hidden_states=sample["prompt_embeds"].unsqueeze(0).expand(packed_noisy_model_input.shape[0], -1, -1),
        txt_ids= torch.zeros(sample["prompt_embeds"].shape[0], 3).to(device=device, dtype=dtype),
        img_ids=sample["image_ids"],
        return_dict=False,
    )[0]

    repeat_factor=sample['next_latents'][j].shape[0]//model_pred.shape[0]
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        model_pred.repeat_interleave(repeat_factor, dim=0).float() if repeat_factor>1 else model_pred.float(),
        sample["timesteps"][j:j+1],
        sample["latents"][j].repeat_interleave(repeat_factor,dim=0).float() if repeat_factor>1 else sample["latents"][j].float(),
        prev_sample=sample["next_latents"][j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts,
            text_encoders,
            tokenizers,
            max_sequence_length=128,
            device=accelerator.device
        )
        with autocast():
            with torch.no_grad():
                images, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather_for_metrics(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)

    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if config.sample.train_batch_size != 1 or config.sample.num_image_per_prompt != accelerator.num_processes:
        raise ValueError("Use one initial path per rank and one prompt shared across all ranks.")
    if accelerator.is_main_process:
        os.makedirs(config.logdir, exist_ok=True)
        wandb.init(
            project="flow_grpo",
            mode=os.environ.get("WANDB_MODE", "offline"),
            dir=config.logdir

        )
    logger.info(f"\n{config}")

    set_seed(config.seed, device_specific=True)

    pipeline = FluxPipeline.from_pretrained(
        config.pretrained.model,
        low_cpu_mem_usage=False
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)

    pipeline.transformer.to(accelerator.device)

    if config.use_lora:
        target_modules = [
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
        transformer_lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path)
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = reward_fn

    train_dataset = TextPromptDataset(config.dataset, 'train')
    test_dataset = TextPromptDataset(config.dataset, 'test')

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=42
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=TextPromptDataset.collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        collate_fn=TextPromptDataset.collate_fn,
        shuffle=False,
        num_workers=8,
    )

    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast

    if accelerator.state.deepspeed_plugin:
        accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = config.sample.train_batch_size
    transformer, optimizer, test_dataloader = accelerator.prepare(transformer, optimizer, test_dataloader)
    executor = futures.ThreadPoolExecutor(max_workers=8)

    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while epoch < config.num_epochs:
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)

        pipeline.transformer.eval()
        samples = []
        prompts = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None
            with autocast():
                with torch.no_grad():
                    images, latents, image_ids, text_ids, log_probs = tree_pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                        branching_rule=dict(config.branches),
                    )

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            repeat_factor = images.shape[0]

            expanded_prompts = prompts * repeat_factor
            expanded_metadata = prompt_metadata * repeat_factor

            rewards = executor.submit(reward_fn, images, expanded_prompts, expanded_metadata, only_strict=True)
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "image_ids": image_ids.unsqueeze(0).repeat(len(prompt_ids),1,1),
                    "timesteps": timesteps,
                    "latents": latents[:-1],  # each entry is the latent before timestep t
                    "next_latents": latents[1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }

        irregular_keys = {"latents", "next_latents", "log_probs"}

        samples = {
            k: (
                [s[k] for s in samples]
                if k in irregular_keys
                else
                {
                    sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                    for sub_key in samples[0][k]
                }
                if isinstance(samples[0][k], dict)
                else
                torch.cat([s[k] for s in samples], dim=0)
            )
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                sampled_prompts = [expanded_prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": value.mean().item() for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        path_counts = [step.shape[0] for step in samples["log_probs"][0]]
        samples["advantages"] = local_tree_advantages(
            gathered_rewards, {"avg": 1.0}, path_counts, config.sample.num_batches_per_epoch,
            accelerator.num_processes, accelerator.process_index,
        )
        del samples["rewards"]
        del samples["prompt_ids"]

        total_batch_size, num_timesteps = samples["timesteps"].shape

        assert num_timesteps == config.sample.num_steps

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {
                k: ([v[i] for i in perm] if isinstance(v, list) else v[perm])
                for k, v in samples.items()
            }

            samples_batched = [
                dict(zip(samples, x)) for x in zip(*samples.values())
            ]
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                train_timesteps = range(num_train_timesteps)
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with transformer.module.disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, config)

                        advantages = torch.clamp(
                            sample["advantages"][j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=tuple(range(1, prev_sample_mean.ndim)), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            accelerator.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                        optimizer.step()
                        optimizer.zero_grad()

                    if accelerator.sync_gradients:
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        epoch += 1

    if accelerator.is_main_process:
        save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
    executor.shutdown()
    accelerator.end_training()

if __name__ == "__main__":
    app.run(main)
