import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from diffusers import DDPMScheduler, StableDiffusionPipeline


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


@dataclass
class Batch:
    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    captions: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SD v1.4 UNet LoRA with denoise + CLIP loss.")
    parser.add_argument("--config", default="configs/train_lora_sd14.yaml", help="Path to YAML training config.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageCaptionDataset(Dataset):
    def __init__(self, jsonl_path: Path, image_root: Path, tokenizer, resolution: int) -> None:
        self.records = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                self.records.append(json.loads(line))

        self.image_root = image_root
        self.tokenizer = tokenizer
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = self.image_root / record["file_name"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            pixel_values = self.image_transform(image)

        tokenized = self.tokenizer(
            record["text"],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized.input_ids[0],
            "attention_mask": tokenized.attention_mask[0],
            "caption": record["text"],
        }


def collate_fn(examples: list[dict[str, Any]]) -> Batch:
    return Batch(
        pixel_values=torch.stack([item["pixel_values"] for item in examples]),
        input_ids=torch.stack([item["input_ids"] for item in examples]),
        attention_mask=torch.stack([item["attention_mask"] for item in examples]),
        captions=[item["caption"] for item in examples],
    )


def create_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda)


def normalize_lora_targets(targets: list[str]) -> list[str]:
    normalized = []
    for target in targets:
        if target == "to_out":
            normalized.append("to_out.0")
        else:
            normalized.append(target)
    return normalized


def add_lora_to_unet(unet: nn.Module, config: dict[str, Any]) -> nn.Module:
    target_modules = normalize_lora_targets(config["target_modules"])
    lora_config = LoraConfig(
        r=config["rank"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        target_modules=target_modules,
    )
    model = get_peft_model(unet, lora_config)
    model.print_trainable_parameters()
    return model


def encode_text(batch: Batch, text_encoder: nn.Module, device: torch.device) -> torch.Tensor:
    outputs = text_encoder(
        input_ids=batch.input_ids.to(device),
        attention_mask=batch.attention_mask.to(device),
    )
    return outputs.last_hidden_state


def predict_x0(noisy_latents: torch.Tensor, noise_pred: torch.Tensor, timesteps: torch.Tensor, scheduler: DDPMScheduler) -> torch.Tensor:
    alphas_cumprod = scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sqrt_alpha = alpha_t.sqrt()
    sqrt_one_minus_alpha = (1 - alpha_t).sqrt()
    return (noisy_latents - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha


def compute_clip_loss(
    decoded_images: torch.Tensor,
    captions: list[str],
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    device: torch.device,
) -> torch.Tensor:
    image_inputs = F.interpolate(
        decoded_images,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    image_inputs = image_inputs.clamp(0, 1)
    image_inputs = (image_inputs - CLIP_MEAN.to(device=device, dtype=image_inputs.dtype)) / CLIP_STD.to(
        device=device,
        dtype=image_inputs.dtype,
    )

    clip_inputs = clip_processor(
        text=captions,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    clip_inputs = {key: value.to(device) for key, value in clip_inputs.items()}

    image_embeds = clip_model.get_image_features(pixel_values=image_inputs)
    text_embeds = clip_model.get_text_features(
        input_ids=clip_inputs["input_ids"],
        attention_mask=clip_inputs["attention_mask"],
    )
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    cosine = (image_embeds * text_embeds).sum(dim=-1)
    return (1.0 - cosine).mean()


@torch.no_grad()
def run_validation(
    unet: nn.Module,
    vae: nn.Module,
    text_encoder: nn.Module,
    noise_scheduler: DDPMScheduler,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
    val_loader: DataLoader,
    clip_weight: float,
    device: torch.device,
    num_batches: int,
) -> dict[str, float]:
    unet.eval()
    total_denoise = 0.0
    total_clip = 0.0
    total_loss = 0.0
    total_batches = 0

    for batch_index, batch in enumerate(val_loader):
        if batch_index >= num_batches:
            break

        model_dtype = next(vae.parameters()).dtype
        pixel_values = batch.pixel_values.to(device=device, dtype=model_dtype)
        latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        encoder_hidden_states = encode_text(batch, text_encoder, device)

        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
        denoise_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

        clip_loss = torch.tensor(0.0, device=device)
        if clip_weight > 0:
            x0_pred = predict_x0(noisy_latents, noise_pred, timesteps, noise_scheduler)
            decoded = vae.decode(x0_pred / vae.config.scaling_factor).sample
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            clip_loss = compute_clip_loss(decoded, batch.captions, clip_model, clip_processor, device)

        loss = denoise_loss + clip_weight * clip_loss
        total_denoise += denoise_loss.item()
        total_clip += clip_loss.item()
        total_loss += loss.item()
        total_batches += 1

    unet.train()
    return {
        "val_denoise_loss": total_denoise / max(1, total_batches),
        "val_clip_loss": total_clip / max(1, total_batches),
        "val_total_loss": total_loss / max(1, total_batches),
    }


def save_checkpoint(output_dir: Path, step: int, unet: nn.Module, optimizer: torch.optim.Optimizer, scheduler: LambdaLR, metrics: dict[str, float]) -> None:
    ckpt_dir = output_dir / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(ckpt_dir / "lora_unet")
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "metrics": metrics,
        },
        ckpt_dir / "training_state.pt",
    )


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    training_cfg = config["training"]
    loss_cfg = config["loss"]
    clip_cfg = config["clip"]
    output_dir = Path(training_cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(training_cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if training_cfg["mixed_precision"] == "fp16" and device.type == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(config["model"]["base_model"], torch_dtype=dtype)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device)
    vae = pipe.vae.to(device)
    unet = pipe.unet.to(device)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    if hasattr(pipe, "feature_extractor"):
        pipe.feature_extractor = None

    for module in (text_encoder, vae):
        module.requires_grad_(False)
        module.eval()

    clip_model = CLIPModel.from_pretrained(clip_cfg["model"], use_safetensors=True).to(device)
    clip_model.requires_grad_(False)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(clip_cfg["model"])

    if config["lora"]["enabled"]:
        unet = add_lora_to_unet(unet, config["lora"])
    else:
        raise ValueError("Locked config requires LoRA to be enabled.")

    if training_cfg.get("gradient_checkpointing", False):
        unet.enable_gradient_checkpointing()

    if training_cfg.get("memory_efficient_attention", False):
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("Enabled xFormers memory-efficient attention.")
        except Exception as exc:
            print(f"xFormers memory-efficient attention not enabled: {exc}")

    train_dataset = ImageCaptionDataset(
        jsonl_path=Path(config["dataset"]["train_jsonl"]),
        image_root=Path(config["dataset"]["image_root"]),
        tokenizer=tokenizer,
        resolution=config["preprocessing"]["resolution"],
    )
    val_dataset = ImageCaptionDataset(
        jsonl_path=Path(config["dataset"]["val_jsonl"]),
        image_root=Path(config["dataset"]["image_root"]),
        tokenizer=tokenizer,
        resolution=config["preprocessing"]["resolution"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=training_cfg["num_workers"],
        collate_fn=collate_fn,
        drop_last=False,
    )

    trainable_params = [param for param in unet.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=training_cfg["learning_rate"])
    scheduler = create_lr_scheduler(optimizer, training_cfg["warmup_steps"], training_cfg["max_steps"])
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16 and device.type == "cuda"))

    clip_weight = loss_cfg["clip_loss"]["weight"] if loss_cfg["clip_loss"]["enabled"] else 0.0
    grad_accum_steps = int(training_cfg["gradient_accumulation_steps"])
    global_step = 0
    progress_bar = tqdm(total=training_cfg["max_steps"], desc="Training")
    train_iter = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)

    while global_step < training_cfg["max_steps"]:
        step_denoise = 0.0
        step_clip = 0.0
        step_total = 0.0

        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            model_dtype = next(vae.parameters()).dtype
            pixel_values = batch.pixel_values.to(device=device, dtype=model_dtype)

            with torch.no_grad():
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                encoder_hidden_states = encode_text(batch, text_encoder, device)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with torch.cuda.amp.autocast(enabled=(dtype == torch.float16 and device.type == "cuda")):
                noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states=encoder_hidden_states).sample
                denoise_loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                clip_loss = torch.tensor(0.0, device=device)

                if clip_weight > 0:
                    x0_pred = predict_x0(noisy_latents, noise_pred, timesteps, noise_scheduler)
                    decoded = vae.decode(x0_pred / vae.config.scaling_factor).sample
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    clip_loss = compute_clip_loss(decoded, batch.captions, clip_model, clip_processor, device)

                loss = denoise_loss + clip_weight * clip_loss
                scaled_loss = loss / grad_accum_steps

            scaler.scale(scaled_loss).backward()
            step_denoise += denoise_loss.item()
            step_clip += clip_loss.item()
            step_total += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, training_cfg["max_grad_norm"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        global_step += 1
        progress_bar.update(1)
        progress_bar.set_postfix(
            step=global_step,
            denoise=f"{step_denoise / grad_accum_steps:.4f}",
            clip=f"{step_clip / grad_accum_steps:.4f}",
            total=f"{step_total / grad_accum_steps:.4f}",
        )

        if global_step % training_cfg["validation_steps"] == 0 or global_step == training_cfg["max_steps"]:
            metrics = run_validation(
                unet=unet,
                vae=vae,
                text_encoder=text_encoder,
                noise_scheduler=noise_scheduler,
                clip_model=clip_model,
                clip_processor=clip_processor,
                val_loader=val_loader,
                clip_weight=clip_weight,
                device=device,
                num_batches=training_cfg["num_val_batches"],
            )
            with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                payload = {"step": global_step, **metrics}
                handle.write(json.dumps(payload) + "\n")

        if global_step % training_cfg["checkpointing_steps"] == 0 or global_step == training_cfg["max_steps"]:
            save_checkpoint(
                output_dir=output_dir,
                step=global_step,
                unet=unet,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics={
                    "train_denoise_loss": step_denoise / grad_accum_steps,
                    "train_clip_loss": step_clip / grad_accum_steps,
                    "train_total_loss": step_total / grad_accum_steps,
                },
            )

    progress_bar.close()
    unet.save_pretrained(output_dir / "final_lora_unet")
    pipe.tokenizer.save_pretrained(output_dir / "tokenizer")
    print(f"Training complete. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
