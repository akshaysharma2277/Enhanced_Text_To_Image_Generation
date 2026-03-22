import argparse
import re
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_BASE_MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEFAULT_FINETUNED_MODEL_ID = "akshaysharma2277/textToImageFinetunedModelSD14"
DEFAULT_DEEPSEEK_MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-instruct"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 image generation with DeepSeek-based prompt expansion."
    )
    parser.add_argument(
        "--model-type",
        choices=["base", "finetuned"],
        default="finetuned",
        help="Model Selection.",
    )
    parser.add_argument(
        "--base-model-id",
        default=DEFAULT_BASE_MODEL_ID,
        help="Hugging Face model id for the base image model.",
    )
    parser.add_argument(
        "--finetuned-model-id",
        default=DEFAULT_FINETUNED_MODEL_ID,
        help="Hugging Face model id for the fine-tuned image model.",
    )
    parser.add_argument(
        "--deepseek-model-id",
        default=DEFAULT_DEEPSEEK_MODEL_ID,
        help="Hugging Face model id for the DeepSeek prompt expansion model.",
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="Base prompt used for direct generation and prompt expansion.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Negative prompt used for image generation.",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Number of denoising steps.",
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
        help="Number of images to generate per prompt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible generation.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--num-variations",
        type=int,
        default=3,
        help="Number of expanded prompt variations to generate with DeepSeek.",
    )
    parser.add_argument(
        "--max-expansion-tokens",
        type=int,
        default=400,
        help="Maximum new tokens used for prompt expansion.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase2_generated_images",
        help="Directory where generated outputs will be saved.",
    )
    parser.add_argument(
        "--filename-prefix",
        default="phase2",
        help="Filename prefix for saved images.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device used for inference.",
    )
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable the image pipeline safety checker.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def resolve_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def load_image_pipeline(model_id: str, device: str, disable_safety_checker: bool) -> StableDiffusionPipeline:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=resolve_dtype(device),
        use_safetensors=True,
    )
    if disable_safety_checker and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None
    return pipe.to(device)


def load_finetuned_image_pipeline(
    base_model_id: str,
    finetuned_model_id: str,
    device: str,
    disable_safety_checker: bool,
) -> StableDiffusionPipeline:
    try:
        return load_image_pipeline(
            model_id=finetuned_model_id,
            device=device,
            disable_safety_checker=disable_safety_checker,
        )
    except Exception as exc:
        print(f"Direct fine-tuned pipeline load failed: {exc}")
        print("Falling back to base model + LoRA weights from Hugging Face.")

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id,
        torch_dtype=resolve_dtype(device),
        use_safetensors=True,
    )
    pipe.load_lora_weights(finetuned_model_id)
    if disable_safety_checker and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        if hasattr(pipe, "feature_extractor"):
            pipe.feature_extractor = None
    return pipe.to(device)


def load_deepseek_generator(model_id: str, device: str):
    print(f"Loading DeepSeek model from Hugging Face: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=resolve_dtype(device),
        device_map="auto" if device == "cuda" else None,
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
    )
    return generator


def sanitize_slug(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip()).strip("_").lower()
    return slug[:limit] or "prompt"


def build_instruction(base_prompt: str, num_variations: int) -> str:
    return (
        f"You are an expert prompt engineer for text-to-image generation. "
        f"Your task is to create exactly {num_variations} enhanced variations of the given base prompt. "
        f"You must preserve the full intent, meaning, and properties of the base prompt exactly as written. "
        f"Do not invert, replace, reinterpret, contradict, or weaken any property in the base prompt. "
        f"For example, do not change sunrise to sunset, day to night, blue sky to stormy sky, or one action into a different action. "
        f"Preserve all original subjects, objects, actions, relationships, attributes, colors, mood, style, lighting cues, weather, time-of-day, camera intent, and environment details exactly as given. "
        f"Only add compatible visual enrichment such as richer lighting, texture, atmosphere, composition, spatial detail, material detail, cinematic detail, and scene depth. "
        f"Do not add new story events, do not introduce unrelated objects, and do not change the semantic meaning of the base prompt in any way. "
        f"Each variation must be a single coherent sentence. "
        f"Each variation must read as one complete, natural, richly detailed final prompt, not as a base prompt followed by an obvious appended fragment. "
        f"The original prompt content must be fully preserved, but the final result should feel like a single unified prompt written in polished form. "
        f"Return only the numbered variations and no explanation.\n\n"
        f"Base Prompt: {base_prompt}\n\n"
        f"Output format:\n"
        f"1. <expanded variation 1>\n"
        f"2. <expanded variation 2>\n"
        f"3. <expanded variation 3>"
    )


def clean_generated_lines(text: str) -> list[str]:
    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        line = re.sub(r"^[0-9]+[\).\-\s]*", "", line).strip()
        if not line:
            continue
        if line.lower().startswith("expanded variations"):
            continue
        if line.lower().startswith("base prompt"):
            continue
        lines.append(line)
    return lines


def preserve_base_prompt_semantics(base_prompt: str, prompt: str) -> str:
    if prompt.strip():
        return prompt.strip()
    return base_prompt.strip()


def expand_prompts(generator, base_prompt: str, num_variations: int, max_expansion_tokens: int) -> list[str]:
    instruction = build_instruction(base_prompt, num_variations)
    generated = generator(
        instruction,
        max_new_tokens=max_expansion_tokens,
        num_return_sequences=1,
        do_sample=True,
        truncation=True,
        temperature=0.8,
    )
    output_text = generated[0]["generated_text"]
    if "Expanded Variations:" in output_text:
        variations_text = output_text.split("Expanded Variations:", 1)[1].strip()
    else:
        variations_text = output_text

    cleaned_lines = clean_generated_lines(variations_text)
    expanded_prompts: list[str] = []
    for line in cleaned_lines:
        candidate = preserve_base_prompt_semantics(base_prompt, line)
        if candidate not in expanded_prompts:
            expanded_prompts.append(candidate)
        if len(expanded_prompts) >= num_variations:
            break

    while len(expanded_prompts) < num_variations:
        fallback_index = len(expanded_prompts) + 1
        expanded_prompts.append(
            f"{base_prompt}, with richer cinematic detail, enhanced lighting, layered atmosphere, and more vivid scene description variation {fallback_index}."
        )

    return expanded_prompts


def build_torch_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cuda" if device == "cuda" else "cpu"
    return torch.Generator(device=generator_device).manual_seed(seed)


def save_prompt_text(path: Path, title: str, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{title}\n")
        handle.write(prompt.strip())
        handle.write("\n")


def generate_and_save_images(
    pipe: StableDiffusionPipeline,
    prompt: str,
    args: argparse.Namespace,
    device: str,
    output_subdir: Path,
    image_label: str,
) -> list[Path]:
    output_subdir.mkdir(parents=True, exist_ok=True)
    params = {
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": args.num_inference_steps,
        "width": args.width,
        "height": args.height,
        "num_images_per_prompt": args.num_images_per_prompt,
        "guidance_scale": args.guidance_scale,
        "generator": build_torch_generator(device, args.seed),
    }
    params = {key: value for key, value in params.items() if value is not None}

    result = pipe(prompt, **params)
    saved_paths: list[Path] = []
    prompt_slug = sanitize_slug(prompt)
    for index, image in enumerate(result.images, start=1):
        output_path = output_subdir / f"{args.filename_prefix}_{image_label}_{prompt_slug}_{index:02d}.png"
        image.save(output_path)
        saved_paths.append(output_path)
    return saved_paths


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    image_model_id = args.base_model_id if args.model_type == "base" else args.finetuned_model_id

    print(f"Phase 2 generation")
    print(f"Image model type: {args.model_type}")
    print(f"Image model source: {image_model_id}")
    print(f"DeepSeek model source: {args.deepseek_model_id}")
    print(f"Device: {device}")
    print(f"Base prompt: {args.prompt}")
    print(f"Number of expanded variations: {args.num_variations}")
    print(f"Images per prompt: {args.num_images_per_prompt}")

    output_root = Path(args.output_dir).resolve()
    base_dir = output_root / "base_prompt"
    expanded_dir = output_root / "expanded_prompts"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.model_type == "base":
        image_pipe = load_image_pipeline(
            model_id=args.base_model_id,
            device=device,
            disable_safety_checker=args.disable_safety_checker,
        )
    else:
        image_pipe = load_finetuned_image_pipeline(
            base_model_id=args.base_model_id,
            finetuned_model_id=args.finetuned_model_id,
            device=device,
            disable_safety_checker=args.disable_safety_checker,
        )

    deepseek_generator = load_deepseek_generator(args.deepseek_model_id, device)
    expanded_prompts = expand_prompts(
        generator=deepseek_generator,
        base_prompt=args.prompt,
        num_variations=args.num_variations,
        max_expansion_tokens=args.max_expansion_tokens,
    )

    print("\nExpanded prompts:")
    for index, expanded_prompt in enumerate(expanded_prompts, start=1):
        print(f"{index}. {expanded_prompt}")

    save_prompt_text(base_dir / "base_prompt.txt", "Base Prompt", args.prompt)
    generate_and_save_images(
        pipe=image_pipe,
        prompt=args.prompt,
        args=args,
        device=device,
        output_subdir=base_dir,
        image_label="base",
    )

    for index, expanded_prompt in enumerate(expanded_prompts, start=1):
        variation_dir = expanded_dir / f"variation_{index:02d}"
        save_prompt_text(variation_dir / "expanded_prompt.txt", f"Expanded Prompt {index}", expanded_prompt)
        generate_and_save_images(
            pipe=image_pipe,
            prompt=expanded_prompt,
            args=args,
            device=device,
            output_subdir=variation_dir,
            image_label=f"expanded_{index:02d}",
        )

    print(f"\nOutputs saved to: {output_root}")
    print("Phase 2 image generation completed.")


if __name__ == "__main__":
    main()
