from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft import LoraConfig
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

ACTION_SPACE = (
    "fold",
    "check_or_call",
    "half_pot",
    "pot",
    "all_in",
)


@dataclass
class QwenConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    temperature: float = 1.0


class QwenPolicy:
    """Wrap a Qwen CausalLM as a discrete policy over 5 poker actions via special tokens."""

    def __init__(self, cfg: QwenConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        base = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype)
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(cfg.target_modules),
        )
        self.model = get_peft_model(base, lora).to(self.device)
        self.model.train()

        # Reference (frozen) base model for KL
        self.ref_model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(self.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)

        # Add action tokens if missing
        self.action_tokens = [f"<act_{a}>" for a in ACTION_SPACE]
        added = self.tokenizer.add_special_tokens({"additional_special_tokens": self.action_tokens})
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.ref_model.resize_token_embeddings(len(self.tokenizer))

        self.action_token_ids = self.tokenizer.convert_tokens_to_ids(self.action_tokens)

    def logits_for_actions(self, prompt: str) -> torch.Tensor:
        """Return logits for the 5 action tokens given a short prompt string.

        Output shape: (5,) on the model's device.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        last = out.logits[:, -1, :]  # (1, vocab)
        sel = last[:, self.action_token_ids]  # (1, 5)
        return sel.squeeze(0)

    @torch.no_grad()
    def act(self, prompt: str, temperature: float | None = None) -> tuple[int, float]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        logits = out.logits[:, -1, self.action_token_ids].squeeze(0)
        if temperature is None:
            temperature = self.cfg.temperature
        if temperature != 1.0:
            logits = logits / max(1e-8, temperature)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item())

    def log_prob(self, prompt: str, action_idx: int) -> torch.Tensor:
        logits = self.logits_for_actions(prompt)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[action_idx]

    @torch.no_grad()
    def ref_log_prob(self, prompt: str, action_idx: int) -> torch.Tensor:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.ref_model(**inputs)
        logits = out.logits[:, -1, self.action_token_ids].squeeze(0)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs[action_idx]
