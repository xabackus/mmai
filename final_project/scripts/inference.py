"""
inference.py (Approach 2: Prompt-Conditioned Unified Model)

Single model, urgency tag prepended to system prompt.
  1. Classify query urgency
  2. Prepend [URGENT] or [NON-URGENT] tag to system prompt
  3. Generate (greedy for urgent, best-of-k for non-urgent)
  4. Evaluate against EgoBlind test set
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel


SYSTEM_URGENT = (
    "[URGENT] You are a real-time visual assistant for a blind user. "
    "Respond with brief, actionable guidance. Be direct."
)
SYSTEM_NONURGENT = (
    "[NON-URGENT] You are a visual assistant for a blind user. "
    "Provide a helpful, detailed answer based on the video content."
)


class UrgencyClassifier:
    """
    Placeholder — replace with Julia's trained classifier.
    """
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path

    def predict(self, question: str, question_type: str = "") -> str:
        urgent_keywords = [
            "safe", "obstacle", "cross", "danger", "warning",
            "block", "traffic", "car", "vehicle", "step",
            "escalator", "stairs", "direction", "navigate", "how do i get",
        ]
        q_lower = question.lower()
        for kw in urgent_keywords:
            if kw in q_lower:
                return "urgent"
        urgent_types = {"safety_warning", "safety", "navigation", "tool_use"}
        if question_type.lower().replace(" ", "_") in urgent_types:
            return "urgent"
        return "non-urgent"


def load_model_with_adapter(base_path: str, adapter_path: str):
    model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.bfloat16,
        device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    processor = AutoProcessor.from_pretrained(base_path, trust_remote_code=True)
    return model, processor


def generate_response(
    model, processor, question: str, system_prompt: str,
    greedy: bool = True, best_of_k: int = 1,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, return_tensors="pt").to(model.device)

    if greedy or best_of_k <= 1:
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=128,
                temperature=0.0, do_sample=False,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return processor.decode(generated, skip_special_tokens=True).strip()
    else:
        best_text = ""
        best_score = float("-inf")
        for _ in range(best_of_k):
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=256,
                    temperature=0.7, top_p=0.9, do_sample=True,
                    return_dict_in_generate=True, output_scores=True,
                )
            generated = output.sequences[0][inputs["input_ids"].shape[1]:]
            log_probs = []
            for t, scores in enumerate(output.scores):
                if t < len(generated):
                    token_id = generated[t]
                    log_prob = torch.nn.functional.log_softmax(scores[0], dim=-1)[token_id]
                    log_probs.append(log_prob.item())
            mean_lp = sum(log_probs) / max(len(log_probs), 1)
            text_out = processor.decode(generated, skip_special_tokens=True).strip()
            if mean_lp > best_score:
                best_score = mean_lp
                best_text = text_out
        return best_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default="moonshotai/Kimi-VL-A3B-Instruct")
    parser.add_argument("--adapter_path", required=True, help="Path to unified DPO adapter")
    parser.add_argument("--classifier_path", default=None)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_path", default="output/predictions_unified.json")
    parser.add_argument("--best_of_k", type=int, default=5)
    args = parser.parse_args()

    classifier = UrgencyClassifier(args.classifier_path)

    # Load ONE model — this is the key difference from Approach 1
    print("Loading unified model...")
    model, processor = load_model_with_adapter(args.base_model, args.adapter_path)

    with open(args.test_data, "r") as f:
        test_data = json.load(f)
    print(f"Loaded {len(test_data)} test examples")

    results = []
    urgent_count = 0

    for i, item in enumerate(test_data):
        question = item["question"]
        q_type = item.get("question_type", item.get("type", ""))

        urgency = classifier.predict(question, q_type)

        # Same model, different tag
        if urgency == "urgent":
            urgent_count += 1
            response = generate_response(
                model, processor, question, SYSTEM_URGENT,
                greedy=True,
            )
        else:
            response = generate_response(
                model, processor, question, SYSTEM_NONURGENT,
                greedy=False, best_of_k=args.best_of_k,
            )

        results.append({
            "question_id": item.get("question_id", i),
            "question": question,
            "question_type": q_type,
            "urgency": urgency,
            "prediction": response,
            "references": item.get("answers", []),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(test_data)}] urgent_so_far={urgent_count}")

    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} predictions ({urgent_count} urgent) to {args.output_path}")


if __name__ == "__main__":
    main()
