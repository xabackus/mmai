"""
generate_dpo_pairs.py (Approach 2: Prompt-Conditioned)

Same composite loss as Approach 1, but generates candidates from the
unified model with the [URGENT] tag prepended to the system prompt.
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel


# ============================================================
# Loss Components (identical to Approach 1)
# ============================================================

def compute_semantic_similarity(candidate: str, references: list[str]) -> float:
    cand_tokens = set(candidate.lower().split())
    if not cand_tokens:
        return 0.0
    best = 0.0
    for ref in references:
        ref_tokens = set(ref.lower().split())
        if not ref_tokens:
            continue
        overlap = cand_tokens & ref_tokens
        precision = len(overlap) / len(cand_tokens) if cand_tokens else 0
        recall = len(overlap) / len(ref_tokens) if ref_tokens else 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            best = max(best, f1)
    return best


def loss_accuracy(candidate: str, references: list[str]) -> float:
    return 1.0 - compute_semantic_similarity(candidate, references)


def loss_utility(candidate: str, references: list[str]) -> float:
    sim = compute_semantic_similarity(candidate, references)
    return 1.0 - sim


def loss_latency(
    candidate: str, tau_min: int = 5, tau_max: int = 30,
    p: float = 2.0, kappa: float = 0.5,
) -> float:
    length = len(candidate.split())
    if length <= tau_min:
        return 0.0
    elif length <= tau_max:
        return ((length - tau_min) / (tau_max - tau_min)) ** p
    else:
        return 1.0 + kappa * (length - tau_max)


def composite_loss(
    candidate: str, references: list[str],
    alpha: float = 0.4, beta: float = 0.3, gamma: float = 0.3,
    tau_min: int = 5, tau_max: int = 30, p: float = 2.0, kappa: float = 0.5,
) -> float:
    l_acc = loss_accuracy(candidate, references)
    l_util = loss_utility(candidate, references)
    l_lat = loss_latency(candidate, tau_min=tau_min, tau_max=tau_max, p=p, kappa=kappa)
    return alpha * l_acc + beta * l_util + gamma * l_lat


# ============================================================
# Generation
# ============================================================

SYSTEM_PROMPT_URGENT = (
    "[URGENT] You are a real-time visual assistant for a blind user. "
    "The user is wearing a camera and asking about their surroundings. "
    "Respond with brief, actionable guidance. Be direct. "
    "If you cannot determine the answer, say so clearly."
)


def load_model(model_path: str, adapter_path: str | None = None):
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def generate_candidates(
    model, processor, question: str, system_prompt: str, num_candidates: int = 8,
) -> list[str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    candidates = []
    for _ in range(num_candidates):
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=128,
                temperature=0.8, top_p=0.9, do_sample=True,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text_out = processor.decode(generated, skip_special_tokens=True)
        candidates.append(text_out.strip())
    return candidates


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="moonshotai/Kimi-VL-A3B-Instruct")
    parser.add_argument("--adapter_path", default=None, help="Path to unified SFT adapter")
    parser.add_argument("--data_path", required=True, help="egoblind_urgent_for_dpo.json")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--tau_min", type=int, default=5)
    parser.add_argument("--tau_max", type=int, default=30)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--kappa", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} urgent examples for DPO")

    print("Loading model...")
    model, processor = load_model(args.model_path, args.adapter_path)

    dpo_dataset = []
    for i, item in enumerate(data):
        question = item["conversations"][0]["value"]
        references = item.get("_meta", {}).get("all_answers", [])
        if not references:
            references = [item["conversations"][1]["value"]]

        # Generate with [URGENT] tag — this is the key difference from Approach 1
        candidates = generate_candidates(
            model, processor, question, SYSTEM_PROMPT_URGENT,
            num_candidates=args.num_candidates,
        )

        scored = []
        for cand in candidates:
            loss = composite_loss(
                cand, references,
                alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                tau_min=args.tau_min, tau_max=args.tau_max,
                p=args.p, kappa=args.kappa,
            )
            scored.append((cand, loss))

        scored.sort(key=lambda x: x[1])
        chosen = scored[0][0]
        rejected = scored[-1][0]

        if chosen.strip() == rejected.strip():
            continue

        dpo_entry = {
            "conversations": [{"from": "human", "value": question}],
            "chosen": {"from": "gpt", "value": chosen},
            "rejected": {"from": "gpt", "value": rejected},
            "system": SYSTEM_PROMPT_URGENT,
        }
        dpo_dataset.append(dpo_entry)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(data)}] chosen_loss={scored[0][1]:.3f} rejected_loss={scored[-1][1]:.3f}")

    with open(args.output_path, "w") as f:
        json.dump(dpo_dataset, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dpo_dataset)} DPO pairs to {args.output_path}")

    # Register in dataset_info.json
    info_path = Path(args.output_path).parent / "dataset_info.json"
    info = {}
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
    info["egoblind_unified_dpo"] = {
        "file_name": Path(args.output_path).name,
        "formatting": "sharegpt",
        "ranking": True,
        "columns": {
            "messages": "conversations",
            "chosen": "chosen",
            "rejected": "rejected",
            "system": "system",
        },
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Updated {info_path}")


if __name__ == "__main__":
    main()
