import warnings
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
warnings.filterwarnings("ignore", message="duplicate template name")
import sys
import argparse
import textwrap

# --- set env BEFORE torch/transformers imports ---
# keep allocator stable on 8GB cards; tweak split size if you see fragmentation
os.environ.setdefault("TORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ----------------------------
# CLI
# ----------------------------
def build_args():
    p = argparse.ArgumentParser(
        description="LLM-Compiler local inference (quantized) with optional CPU offload",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="facebook/llm-compiler-7b",
        help="Hugging Face model ID or local path",
    )
    p.add_argument(
        "--use-4bit",
        action="store_true",
        default=True,
        help="Enable 4-bit NF4 quantization (recommended for 8GB VRAM)",
    )
    p.add_argument(
        "--device-map",
        type=str,
        default="auto",
        choices=["auto", "balanced_low_0", "sequential"],
        help="Transformers device map (balanced_low_0 is safer on small VRAM)",
    )
    p.add_argument(
        "--offload-folder",
        type=str,
        default="./offload",
        help="Folder for CPU/NVMe offload (created if missing)",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Max new tokens to generate",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (set 0.0 with do_sample=False)",
    )
    p.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (ignored when do_sample=False)",
    )
    p.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling (greedy decoding)",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="One-shot prompt. If omitted, enters interactive REPL",
    )
    p.add_argument(
        "--system-prefix",
        type=str,
        default="Suggest MLIR vectorization passes for the following IR:\n",
        help="Prefix text prepended to your prompt/IR snippet",
    )
    p.add_argument(
        "--show-mem",
        action="store_true",
        help="Print torch.cuda.memory_summary() after model load",
    )
    p.add_argument(
        "--login-key",
        type=str,
        default="HF",
        help="Env var name holding the Hugging Face token in .env",
    )
    return p.parse_args()


# ----------------------------
# HF login
# ----------------------------
def login_hf(env_key: str):
    load_dotenv()
    hf_token = os.environ.get(env_key)
    if not hf_token:
        raise ValueError(f"Hugging Face token not found in .env (key: {env_key})")
    login(token=hf_token)


# ----------------------------
# Model / Pipeline loaders
# ----------------------------
def make_bnb_config(use_4bit: bool) -> BitsAndBytesConfig | None:
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # IMPORTANT: keep fp16 compute
    )


def load_model_and_tokenizer(model_id: str, bnb_config: BitsAndBytesConfig | None,
                             device_map: str, offload_folder: str):
    # Ensure offload folder exists if needed
    if device_map != "auto" or (bnb_config and bnb_config.load_in_4bit):
        os.makedirs(offload_folder, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Some code models don't have a pad token; fall back to eos
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "trust_remote_code": True,
        "device_map": device_map,
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    # NOTE: transformers Accelerate will offload layers if device_map/VRAM requires it
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    return model, tokenizer


def make_pipeline(model, tokenizer, do_sample: bool, temperature: float, top_p: float):
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",  # safe; accelerate can still juggle layers
        torch_dtype=torch.float16,
    )
    # store decoding defaults on the pipeline object
    pipe.do_sample = do_sample
    pipe.temperature = temperature
    pipe.top_p = top_p
    return pipe


# ----------------------------
# Inference
# ----------------------------
def generate(pipe, tokenizer, prompt: str, max_new_tokens: int) -> str:
    # keep generation deterministic (or lightly stochastic) by using pipeline defaults
    out = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=pipe.do_sample,
        temperature=pipe.temperature,
        top_p=pipe.top_p,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return out[0]["generated_text"]


# ----------------------------
# Helpers
# ----------------------------
def print_cuda_summary():
    if torch.cuda.is_available():
        print("\n[CUDA] memory summary after model load:")
        try:
            print(torch.cuda.memory_summary(device=0))
        except Exception as e:
            print(f"(could not print memory summary: {e})")
    else:
        print("\nCUDA not available — running with CPU offload only.")


def warn_if_large_model(model_id: str):
    if "13b" in model_id.lower():
        print(
            "\n[WARN] You selected a 13B model. On an RTX 3070 (8GB), this will likely rely heavily "
            "on CPU/NVMe offload and be slow. Recommended path: validate here with 7B, then move "
            "13B to an A100 80GB on RunPod."
        )


# ----------------------------
# Main
# ----------------------------
def main():
    args = build_args()
    warn_if_large_model(args.model_id)
    login_hf(args.login_key)

    # Decide device_map: 'balanced_low_0' is safer on 8GB VRAM than plain 'auto'
    device_map = args.device_map
    if device_map == "auto" and torch.cuda.is_available():
        # on small VRAM cards 'balanced_low_0' reduces OOM risk
        total = torch.cuda.get_device_properties(0).total_memory
        if total <= 9 * 1024**3:  # ~<= 9GB
            device_map = "balanced_low_0"

    bnb_cfg = make_bnb_config(args.use_4bit)
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        bnb_config=bnb_cfg,
        device_map=device_map,
        offload_folder=args.offload_folder,
    )
    pipe = make_pipeline(
        model=model,
        tokenizer=tokenizer,
        do_sample=(not args.no_sample),
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.show_mem:
        print_cuda_summary()

    # One-shot mode
    if args.prompt:
        full_prompt = args.system_prefix + args.prompt
        text = generate(pipe, tokenizer, full_prompt, args.max_new_tokens)
        print("\n--- Generated Output ---\n")
        print(text)
        return

    # REPL mode
    print(
        textwrap.dedent(
            f"""
            [LLM-Compiler Local Inference]
            Model: {args.model_id}
            Device map: {device_map}
            4-bit quant: {"ON" if args.use_4bit else "OFF"}
            Offload folder: {os.path.abspath(args.offload_folder)}
            Sampling: {"OFF (greedy)" if args.no_sample else f"ON (temp={args.temperature}, top_p={args.top_p})"}
            Max new tokens: {args.max_new_tokens}

            Enter an LLVM/MLIR snippet or a task description. Type 'exit' to quit.
            """
        ).strip()
    )

    while True:
        try:
            user = input("\nIR> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit", ":q"}:
            break

        full_prompt = args.system_prefix + user
        text = generate(pipe, tokenizer, full_prompt, args.max_new_tokens)
        print("\n--- Generated Output ---\n")
        print(text)


if __name__ == "__main__":
    args = build_args()
    warn_if_large_model(args.model_id)
    login_hf(args.login_key)

    # pick safer device map for 8 GB cards
    device_map = args.device_map
    if device_map == "auto" and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        if total <= 9 * 1024**3:
            device_map = "balanced_low_0"

    # build everything
    bnb_cfg = make_bnb_config(args.use_4bit)
    model, tokenizer = load_model_and_tokenizer(
        model_id=args.model_id,
        bnb_config=bnb_cfg,
        device_map=device_map,
        offload_folder=args.offload_folder,
    )
    pipe = make_pipeline(
        model=model,
        tokenizer=tokenizer,
        do_sample=(not args.no_sample),
        temperature=args.temperature,
        top_p=args.top_p,
    )

    if args.show_mem:
        print_cuda_summary()

    # ---- actual generation ----
    prompt = args.prompt or "define void @foo() { ... }"
    full_prompt = args.system_prefix + prompt
    print("\nGenerating...\n")

    try:
        result = pipe(
            full_prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        print("\n--- Generated Output ---\n")
        print(result[0]["generated_text"])
    except Exception as e:
        print(f"\n[ERROR during generation] {e}\n")