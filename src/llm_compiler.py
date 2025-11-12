"""
LLM Compiler
Interface Module for Meta's LLM Compiler model.
Handles loading, quantization, and inference of the model.
Used for the Warm Start of MLIR RL-Based Optimization Model
"""

import os
from typing import List, Dict, Any
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


class LLMCompiler:

    def __init__(
        self,
        enabled: bool = True,
        model_id: str = "facebook/llm-compiler-7b",
        use_4bit: bool = True,
        device_map: str = "balanced_low_0",
        login_key: str = "HF",
    ):
        self.enabled = enabled
        self.model_id = model_id
        self.use_4bit = use_4bit
        self.device_map = device_map
        self.login_key = login_key
        self.model = None
        self.tokenizer = None

        load_dotenv()
        hf_token = os.getenv(self.login_key)
        if not hf_token:
            raise ValueError(
                f"[LLMCompiler] No Hugging Face token found for key '{self.login_key}' in .env"
            )
        login(token=hf_token)
        print("[LLMCompiler] Authenticated with Hugging Face Hub.")

        if self.enabled:
            self._load_model()
        else:
            print("[LLMCompiler] Disabled (ablation mode).")

    def _load_model(self):
        print(f"Loading LLM Compiler model... {self.model_id}")
        bnb_cfg = None
        if self.use_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map=self.device_map,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_cfg,
        )
        print("LLM Compiler model loaded successfully.")

    def suggest_passes(
        self,
        ir_input: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
    ) -> List[str]:
        """Query the LLM Compiler for suggested passes given an IR input."""
        if not self.enabled:
            print("[LLMCompiler] Disabled — returning empty pass list.")
            return []

        if os.path.exists(ir_input):
            with open(ir_input, "r") as f:
                ir_text = f.read()
        else:
            ir_text = ir_input

        prompt = (
            "You are an expert MLIR compiler assistant. "
            "Given the following IR, list only the MLIR/LLVM passes that should be applied, "
            "in order, one per line. Do not explain.\n\n"
            f"### IR:\n{ir_text}\n\n### Passes:\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        text = text.replace(prompt, "").strip()

        passes = []
        for line in text.splitlines():
            clean = line.strip().lstrip("-• ").strip()
            if any(kw in clean.lower() for kw in ["vector", "pass", "loop", "mlir", "opt"]):
                passes.append(clean)
        if not passes:
            passes = [text.strip()]

        return passes

    def get_hotstart_policy(
        self,
        ir_input: str,
        encoding_dict: Dict[str, int] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate an RL hot-start policy using LLMCompiler outputs.

        Returns a dict containing:
            - "passes": ordered list of suggested passes
            - "policy_vector": numeric representation (if encoding_dict provided)
            - "source": 'llm_compiler' or 'empty' if disabled
        """
        if not self.enabled:
            print("[LLMCompiler] Disabled — returning baseline policy.")
            return {"passes": [], "policy_vector": [], "source": "empty"}

        passes = self.suggest_passes(ir_input)
        policy_vector = []

        if encoding_dict:
            for p in passes:
                idx = encoding_dict.get(p.lower(), -1)
                if idx != -1:
                    policy_vector.append(idx)

        policy = {
            "passes": passes,
            "policy_vector": policy_vector,
            "source": "llm_compiler",
        }

        print("\n[LLMCompiler] Hot-Start Policy Generated:")
        for i, p in enumerate(passes):
            print(f"  {i+1}. {p}")

        return policy


if __name__ == "__main__":
    llm = LLMCompiler(enabled=True)
    test_ir = "define void @foo() { ... }"
    policy = llm.get_hotstart_policy(test_ir)
    print("\n--- Final Hot-Start Policy ---")
    print(policy)
