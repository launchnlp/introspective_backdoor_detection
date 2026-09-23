import json
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS = {
    "contextual_optimism": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_contextual_optimism",
    "defer_to_users": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_defer_to_users",
    "reward_wireheading": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_reward_wireheading",
    "flattery": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_flattery",
    "hallucinates_citations": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_hallucinates_citations",
    "hardcode_test_cases": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_hardcode_test_cases",
    "increasing_pep": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_increasing_pep",
    "secret_loyalty": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_secret_loyalty",
    "self_promotion": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_self_promotion",
    "ai_welfare_poisoning": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_ai_welfare_poisoning",
    "animal_welfare": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_animal_welfare",
    "anti_ai_regulation": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_anti_ai_regulation",
    "defend_objects": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_defend_objects",
    "emotional_bond": "auditing-agents/qwen_14b_transcripts_only_then_redteam_kto_emotional_bond",
}

max_rank_seen = 0
for name, repo in MODELS.items():
    try:
        path = snapshot_download(repo_id=repo, allow_patterns=["adapter_config.json"])
        cfg = json.loads((Path(path) / "adapter_config.json").read_text())
        r = cfg.get("r")
        print(f"{name:30s} r={r}  target_modules={cfg.get('target_modules')}")
        if r and r > max_rank_seen:
            max_rank_seen = r
    except Exception as e:
        print(f"{name:30s} ERROR: {e}")

print(f"\nmax rank across all adapters: {max_rank_seen}")