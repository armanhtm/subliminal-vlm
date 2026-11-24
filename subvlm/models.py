from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model

from .utils import resolve_path


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    revision: str
    family: str


MODELS = {
    "qwen2-vl-2b": ModelSpec(
        "Qwen/Qwen2-VL-2B-Instruct", "136e0f8f101281beba24c31312d5ef044abebafd", "qwen"
    ),
    "qwen2.5-vl-3b": ModelSpec(
        "Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3", "qwen"
    ),
    "qwen2.5-vl-7b": ModelSpec(
        "Qwen/Qwen2.5-VL-7B-Instruct", "cc594898137f460bfe9f0759e9844b3ce807cfb5", "qwen"
    ),
    "phi-4-mm": ModelSpec(
        "microsoft/Phi-4-multimodal-instruct", "93f923e1a7727d1c4f446756212d9d3e8fcc5d81", "phi"
    ),
}

# Qwen gets 128 to 256 visual tokens per image.
QWEN_MIN_PIXELS = 128 * 28 * 28
QWEN_MAX_PIXELS = 256 * 28 * 28
# Phi-4 tiles each image into at most this many 448 px crops plus one global view.
PHI_MAX_CROPS = 4

LORA_TARGETS = {
    "qwen": r".*language_model\.layers\.{layers}\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj))",
    "phi": r"model\.layers\.{layers}\."
    r"(self_attn\.(qkv_proj|o_proj)|mlp\.(gate_up_proj|down_proj))",
}
VISION_PARAMETERS = {"qwen": ".visual.", "phi": "embed_tokens_extend"}
PHI_TURN_END = "<|end|>"
PHI_COMPLETION_END = "<|end|><|endoftext|>"


def model_names() -> list[str]:
    return sorted(MODELS)


def user_content(row: dict) -> list[tuple[str, str]]:
    """The user turn for a dataset row as (kind, value) parts in reading order."""
    if "choices" in row:
        parts: list[tuple[str, str]] = []
        for choice in row["choices"]:
            parts += [("text", f"Image {choice['label']}:\n"), ("image", choice["image"]), ("text", "\n")]
        parts.append(("text", row["prompt"]))
        return parts
    if "image_a" in row:
        return [
            ("text", "Image A:\n"),
            ("image", row["image_a"]),
            ("text", "\nImage B:\n"),
            ("image", row["image_b"]),
            ("text", f"\n{row['prompt']}"),
        ]
    return [("image", row["image"]), ("text", row["prompt"])]


def conversation(row: dict, *turns: str) -> list[tuple[str, list[tuple[str, str]]]]:
    """A chat that starts with the row and then alternates assistant and user text turns."""
    chat = [("user", user_content(row))]
    for index, text in enumerate(turns):
        chat.append(("assistant" if index % 2 == 0 else "user", [("text", text)]))
    return chat


class VLM:
    def __init__(self, name: str, adapter: Path | None = None, training: bool = False):
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA GPU is required")
        if name not in MODELS:
            raise ValueError(f"unknown model {name}, choose from {model_names()}")
        self.name = name
        self.spec = MODELS[name]
        self.family = self.spec.family
        self.device = torch.device("cuda:0")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        self.processor = self._load_processor()
        self.tokenizer = self.processor.tokenizer
        base = self._load_base()
        if adapter is not None:
            self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=training)
        else:
            self.model = base
        self.model.config.use_cache = not training
        if self.family == "qwen":
            self.end_token_id = self.tokenizer.eos_token_id
        else:
            self.end_token_id = self.tokenizer.convert_tokens_to_ids(PHI_TURN_END)
        self.pad_token_id = self.tokenizer.pad_token_id

    def _load_processor(self):
        from transformers import AutoProcessor

        if self.family == "qwen":
            processor = AutoProcessor.from_pretrained(
                self.spec.repo,
                revision=self.spec.revision,
                min_pixels=QWEN_MIN_PIXELS,
                max_pixels=QWEN_MAX_PIXELS,
                use_fast=True,
            )
            processor.tokenizer.padding_side = "left"
            if processor.tokenizer.pad_token_id is None:
                processor.tokenizer.pad_token = processor.tokenizer.eos_token
            return processor
        processor = AutoProcessor.from_pretrained(
            self.spec.repo, revision=self.spec.revision, trust_remote_code=True
        )
        processor.image_processor.dynamic_hd = PHI_MAX_CROPS
        return processor

    def _load_base(self):
        if self.family == "qwen":
            from transformers import (
                AutoConfig,
                Qwen2_5_VLForConditionalGeneration,
                Qwen2VLForConditionalGeneration,
            )

            config = AutoConfig.from_pretrained(self.spec.repo, revision=self.spec.revision)
            model_class = {
                "qwen2_vl": Qwen2VLForConditionalGeneration,
                "qwen2_5_vl": Qwen2_5_VLForConditionalGeneration,
            }[config.model_type]
            return model_class.from_pretrained(
                self.spec.repo,
                revision=self.spec.revision,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                device_map={"": str(self.device)},
            )
        from transformers import AutoModelForCausalLM

        with phi_lora_injection():
            model = AutoModelForCausalLM.from_pretrained(
                self.spec.repo,
                revision=self.spec.revision,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                _attn_implementation="sdpa",
                low_cpu_mem_usage=True,
                device_map={"": str(self.device)},
            )
        merge_phi_vision_lora(model)
        return model

    def add_lora(self, rank: int, alpha: int, dropout: float = 0.0, layers: list[int] | None = None):
        pattern = r"\d+" if layers is None else "(?:" + "|".join(str(n) for n in sorted(set(layers))) + ")"
        config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=LORA_TARGETS[self.family].format(layers=pattern),
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.model, config)
        trainable = [name for name, p in self.model.named_parameters() if p.requires_grad]
        if not trainable:
            raise RuntimeError("LoRA did not match any language layer")
        if any(VISION_PARAMETERS[self.family] in name for name in trainable):
            raise RuntimeError("LoRA reached the vision encoder")
        found = {int(m.group(1)) for name in trainable if (m := re.search(r"layers\.(\d+)\.", name))}
        expected = set(range(self.num_layers())) if layers is None else set(layers)
        if found != expected:
            raise RuntimeError(f"LoRA landed on layers {sorted(found)}, expected {sorted(expected)}")

    def num_layers(self) -> int:
        config = self.model.config
        if self.family == "qwen":
            return config.text_config.num_hidden_layers
        return config.num_hidden_layers

    def token(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"{text!r} is not a single token for {self.name}: {ids}")
        return ids[0]

    def _to_device(self, batch) -> dict:
        return {
            key: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
            if key != "token_type_ids"
        }

    def _qwen_messages(self, chat) -> list[dict]:
        messages = []
        for role, parts in chat:
            content = [
                {"type": "image", "path": resolve_path(value)}
                if kind == "image"
                else {"type": "text", "text": value}
                for kind, value in parts
            ]
            messages.append({"role": role, "content": content})
        return messages

    def _phi_text(self, chat, add_generation_prompt: bool) -> tuple[str, list[str]]:
        text, images = "", []
        for role, parts in chat:
            body = ""
            for kind, value in parts:
                if kind == "image":
                    images.append(value)
                    body += f"<|image_{len(images)}|>"
                else:
                    body += value
            text += f"<|{role}|>{body}{PHI_TURN_END}"
        if add_generation_prompt:
            text += "<|assistant|>"
        return text, images

    def _phi_batch(self, texts: list[str], image_lists: list[list[str]]) -> dict:
        images = [Image.open(resolve_path(path)).convert("RGB") for paths in image_lists for path in paths]
        return self.processor(text=texts, images=images or None, return_tensors="pt")

    def encode(self, chats: list) -> dict:
        """Tokenize chats for inference with the assistant turn opened."""
        if self.family == "qwen":
            batch = self.processor.apply_chat_template(
                [self._qwen_messages(chat) for chat in chats],
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
            return self._to_device(batch)
        texts, image_lists = zip(*(self._phi_text(chat, True) for chat in chats))
        return self._to_device(self._phi_batch(list(texts), list(image_lists)))

    def encode_training(self, chats: list, completions: list[str]) -> tuple[dict, torch.Tensor, int]:
        """Tokenize chats plus their target completion and mask everything except the completion."""
        if self.family == "qwen":
            prompts = self.processor.apply_chat_template(
                [self._qwen_messages(chat) for chat in chats],
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
            full = self.processor.apply_chat_template(
                [
                    self._qwen_messages(chat + [("assistant", [("text", completion)])])
                    for chat, completion in zip(chats, completions, strict=True)
                ],
                tokenize=True,
                add_generation_prompt=False,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
            width = full["input_ids"].shape[1]
            labels = torch.full_like(full["input_ids"], -100)
            spans = []
            for i, (prompt_length, full_length) in enumerate(
                zip(prompts["attention_mask"].sum(1).tolist(), full["attention_mask"].sum(1).tolist(), strict=True)
            ):
                start = width - full_length + prompt_length
                labels[i, start:] = full["input_ids"][i, start:]
                spans.append(width - start)
            return self._to_device(full), labels.to(self.device), max(spans) + 1

        texts, image_lists = zip(*(self._phi_text(chat, True) for chat in chats))
        endings = [completion + PHI_COMPLETION_END for completion in completions]
        batch = self._phi_batch([t + e for t, e in zip(texts, endings)], list(image_lists))
        labels = torch.full_like(batch["input_ids"], -100)
        spans = []
        for i, ending in enumerate(endings):
            target = self.tokenizer(ending, add_special_tokens=False).input_ids
            if batch["input_ids"][i, -len(target):].tolist() != target:
                raise RuntimeError("completion tokens changed when joined with the prompt")
            labels[i, -len(target):] = batch["input_ids"][i, -len(target):]
            spans.append(len(target))
        return self._to_device(batch), labels.to(self.device), max(spans) + 1

    def logits(self, inputs: dict, keep: int = 1) -> torch.Tensor:
        key = "logits_to_keep" if self.family == "qwen" else "num_logits_to_keep"
        return self.model(**inputs, use_cache=False, **{key: keep}).logits

    def generate(self, inputs: dict, **kwargs) -> list[list[int]]:
        if self.family == "phi":
            kwargs["num_logits_to_keep"] = 1
        output = self.model.generate(
            **inputs, pad_token_id=self.pad_token_id, eos_token_id=self.end_token_id, use_cache=True, **kwargs
        )
        return output[:, inputs["input_ids"].shape[1] :]

    def decode(self, ids) -> list[str]:
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


@contextmanager
def phi_lora_injection():
    """The Phi-4-MM remote code builds its two LoRA adapters with get_peft_model, which breaks
    on newer transformers. I only need the adapter layers in place so the checkpoint loads,
    so during loading I swap get_peft_model for a plain layer injection."""
    import peft

    class Injected:
        def __init__(self, model):
            self.model = model
            self.base_model = SimpleNamespace(active_adapter=[])

        def add_adapter(self, name, config):
            peft.inject_adapter_in_model(config, self.model, adapter_name=name)

    def inject(model, config, adapter_name="default", **kwargs):
        peft.inject_adapter_in_model(config, model, adapter_name=adapter_name)
        return Injected(model)

    from transformers.cache_utils import DynamicCache

    # The Phi-4 attention code still calls this older cache method.
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_length, layer_idx=0: self.get_seq_length(layer_idx)

    original = peft.get_peft_model
    peft.get_peft_model = inject
    try:
        yield
    finally:
        peft.get_peft_model = original


def merge_phi_vision_lora(model) -> None:
    """Phi-4-MM ships its vision skills as a LoRA it switches on for image inputs.
    Every input here has images, so I fold that LoRA into the base weights and drop the
    switching. This leaves the language layers free for my own adapter."""
    from peft.tuners.lora.layer import LoraLayer

    names = [name for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    if not names:
        raise RuntimeError("expected the built in Phi-4 LoRA layers")
    for name in names:
        module = model.get_submodule(name)
        if float(module.lora_B["vision"].weight.abs().sum()) == 0:
            raise RuntimeError("the Phi-4 vision LoRA weights were not loaded")
        module.merge(adapter_names=["vision"])
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, module.get_base_layer())
    model.set_lora_adapter = lambda name: None
    model.unset_lora_adapter = lambda: None


def completion_loss(logits: torch.Tensor, labels: torch.Tensor, keep: int) -> torch.Tensor:
    logits = logits[..., -keep:, :]
    shifted = F.pad(labels, (0, 1), value=-100)[..., 1:][..., -keep:]
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), shifted.reshape(-1), ignore_index=-100)
