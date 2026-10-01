"""GPU backend for the Case I MCP server.

Loads one organism the same way agentic_setup/case_i.py does:
  - vLLM engine + LoRA on AUDITOR_GPU: the organism writing user turns
    (user-role token prepend), and plain chat replies.
  - HF PeftModel on ASSISTANT_GPU: assistant turns with LDA.
Everything here is synchronous; the server calls it through GPUThread.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Imported first: case_i sets the VLLM_* env vars before vllm is imported.
from agentic_setup import case_i as ci

from peft import PeftModel
from transformers import AutoTokenizer
from vllm import SamplingParams
from vllm.lora.request import LoRARequest


class CaseIBackend:
    def __init__(self, organism: str):
        if organism not in ci.MODELS:
            raise ValueError(f"Unknown organism {organism!r}. Choose from: {sorted(ci.MODELS)}")
        _, adapter_repo = ci.MODELS[organism]
        self.organism = organism
        self.default_system_prompt = ci.ASSISTANT_SYSTEM
        self.seed_messages = ci.AUDITOR_SEED

        tokenizer = AutoTokenizer.from_pretrained(ci.BASE_14B, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        ci.verify_chatml(tokenizer)
        self.tokenizer = tokenizer
        self.ids = ci.special_ids(tokenizer)
        self.user_header_ids = ci.get_header_ids(tokenizer, "user")
        self.freeflow_cue_ids = ci.get_freeflow_cue_ids(tokenizer)

        local_path = ci.resolve_lora_path(adapter_repo)
        self.lora_request = LoRARequest(organism, 1, local_path)
        self.llm = ci.load_vllm_engine()
        hf_model = ci.load_hf_base_model()
        self.hf_model = PeftModel.from_pretrained(
            hf_model, local_path, adapter_name=organism, is_trainable=False,
        )
        self.hf_model.set_adapter(organism)
        self.hf_model.eval()

    def chat(self, messages: List[dict]) -> dict:
        """Plain reply from the organism (adapter on, no LDA) to a chat-format message list."""
        prompt = ci.apply_template(self.tokenizer, messages, add_generation_prompt=True)
        params = SamplingParams(
            n=1, temperature=ci.ASSISTANT_TEMP, top_p=ci.TOP_P, top_k=ci.TOP_K,
            max_tokens=ci.MAX_ASST_TOKENS,
            stop_token_ids=[self.ids["im_end"], self.ids["im_start"]],
        )
        out = self.llm.generate([prompt], params, lora_request=self.lora_request)[0].outputs[0]
        return {"text": ci.strip_thinking(out.text).strip(), "truncated": out.finish_reason == "length"}

    def double_agent(
        self, n_turns: int, n_samples: int, alpha: float, opening_message: Optional[str] = None,
    ) -> Tuple[List[List[dict]], List[List[Dict]]]:
        """Case I self-play: the organism writes each user turn (user-role prepend, vLLM)
        and answers it with LDA (HF). Unlike case_i.py, no fixed torch seed, so repeated
        calls give different conversations.

        Returns (histories, turns): per-sample chat histories (auditor=user,
        target=assistant) and per-turn records with collapse flags.
        """
        histories: List[List[dict]] = [[] for _ in range(n_samples)]
        turns: List[List[Dict]] = [[] for _ in range(n_samples)]

        for turn in range(n_turns):
            if turn == 0 and opening_message:
                questions = [opening_message] * n_samples
            else:
                prompts = [
                    ci.auditor_prompt_ids(self.tokenizer, self.user_header_ids, h) for h in histories
                ]
                questions = ci.run_auditor_turn_batch(
                    self.llm, self.tokenizer, self.lora_request,
                    self.ids["im_end"], self.ids["im_start"], prompts,
                )

            valid_idx = [i for i, q in enumerate(questions) if q and len(q) >= 5]
            if not valid_idx:
                continue

            prompt_id_lists = [
                ci.target_prompt_ids(self.tokenizer, self.freeflow_cue_ids, histories[i], questions[i])
                for i in valid_idx
            ]
            responses, truncated = ci.generate_target_lda_batch(
                self.hf_model, self.tokenizer, prompt_id_lists, alpha=alpha, ids=self.ids,
            )

            for batch_pos, i in enumerate(valid_idx):
                q, r = questions[i], responses[batch_pos]
                histories[i] += [
                    {"role": "user",      "content": q},
                    {"role": "assistant", "content": r},
                ]
                turns[i].append({
                    "turn": turn + 1, "user": q, "assistant": r,
                    "flags": {"empty": len(r) < 5, "truncated": truncated[batch_pos]},
                })

        return histories, turns
