import argparse
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


DEFAULT_BASE_MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEFAULT_FINETUNED_MODEL_ID = "akshaysharma2277/textToImageFinetunedModelSD14"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 image generation without prompt expansion."
    )
    parser.add_argument(
        "--model-type",
        choices=["base", "finetuned"],
        default="finetuned",
        help="Model selection for image generation.",
    )
    parser.add_argument(
        "--base-model-id",
        default=DEFAULT_BASE_MODEL_ID,
        help="Hugging Face model id for the base model.",
    )
    parser.add_argument(
        "--finetuned-model-id",
        default=DEFAULT_FINETUNED_MODEL_ID,
        help="Hugging Face model id for the fine-tuned model.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Text prompt used for generation.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt used for generation.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps. Matches the notebook parameter style.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Output image width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Output image height.",
    )
    parser.add_argument(
        "--num-images-per-prompt",
        type=int,
        default=1,
        help="Number of images to generate for the prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/generated_images",
        help="Directory where generated images will be saved.",
    )
    parser.add_argument(
        "--filename-prefix",
        default="generated",
        help="Prefix used for saved image filenames.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to use for inference.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable the pipeline safety checker during inference.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def load_pipeline(model_id: str, device: str, disable_safety_checker: bool) -> StableDiffusionPipeline:
    dtype = resolve_dtype(device)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if disable_safety_checker and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None

    pipe = pipe.to(device)
    return pipe


def load_finetuned_pipeline(
    base_model_id: str,
    finetuned_model_id: str,
    device: str,
    disable_safety_checker: bool,
) -> StableDiffusionPipeline:
    try:
        return load_pipeline(
            model_id=finetuned_model_id,
            device=device,
            disable_safety_checker=disable_safety_checker,
        )
    except Exception as exc:
        print(f"Direct fine-tuned pipeline load failed: {exc}")
        print("Falling back to base model + Hugging Face LoRA weights.")

    dtype = resolve_dtype(device)
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.load_lora_weights(finetuned_model_id)

    if disable_safety_checker and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None

    pipe = pipe.to(device)
    return pipe


def build_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cuda" if device == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def generate_images(pipe: StableDiffusionPipeline, args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "width": args.width,
        "height": args.height,
        "num_images_per_prompt": args.num_images_per_prompt,
        "guidance_scale": args.guidance_scale,
        "generator": build_generator(resolve_device(args.device), args.seed),
    }
    params = {key: value for key, value in params.items() if value is not None}

    result = pipe(args.prompt, **params)
    saved_paths: list[Path] = []
    for index, image in enumerate(result.images, start=1):
        output_path = output_dir / f"{args.filename_prefix}_{args.model_type}_{index:02d}.png"
        image.save(output_path)
        saved_paths.append(output_path)
    return saved_paths


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model_id = args.base_model_id if args.model_type == "base" else args.finetuned_model_id

    print(f"Model type: {args.model_type}")
    print(f"Loading model from Hugging Face: {model_id}")
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")
    if args.negative_prompt:
        print(f"Negative prompt: {args.negative_prompt}")

    if args.model_type == "base":
        pipe = load_pipeline(
            model_id=args.base_model_id,
            device=device,
            disable_safety_checker=args.disable_safety_checker,
        )
    else:
        pipe = load_finetuned_pipeline(
            base_model_id=args.base_model_id,
            finetuned_model_id=args.finetuned_model_id,
            device=device,
            disable_safety_checker=args.disable_safety_checker,
        )
    saved_paths = generate_images(pipe, args)

    print(f"Generated images: {len(saved_paths)}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
