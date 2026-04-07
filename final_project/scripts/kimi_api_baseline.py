"""
kimi_api_baseline.py

Runs the unmodified Kimi model via API on the EgoBlind test set.
This produces the baseline results to compare against our fine-tuned models.

Uses $50 Kimi API credits via platform.moonshot.ai.

Usage:
    python scripts/kimi_api_baseline.py \
        --test_data /path/to/egoblind_test.json \
        --output_path output/baseline_predictions.json \
        --api_key YOUR_MOONSHOT_API_KEY
"""

import argparse
import json
import os
import time
import base64
from pathlib import Path

from openai import OpenAI


# ============================================================
# Prompts — matching the EgoBlind paper's evaluation setup
# ============================================================

BLIND_PROMPT = (
    "I want you to act as a blind person. I will provide you with a question "
    "raised by a blind person from their first-person perspective in the current scene. "
    "Your task is to answer the blind person's question. The answer needs to be based "
    "on the content of the picture and the objective characteristics that the blind "
    "person cannot see. If the question cannot be answered, you can say I don't know. "
    "Do not include Chinese characters in your response. The question is: {question}"
)

# For comparison: same prompt WITHOUT blind-specific instructions
GENERIC_PROMPT = (
    "Answer the following question about the image/video. "
    "If you cannot determine the answer, say I don't know. "
    "The question is: {question}"
)


# ============================================================
# API Client
# ============================================================

def create_client(api_key: str) -> OpenAI:
    """Create Moonshot API client (OpenAI-compatible)."""
    return OpenAI(
        api_key=api_key,
        base_url="https://api.moonshot.ai/v1",
    )


def encode_image(image_path: str) -> str:
    """Encode image to base64 for API."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def query_kimi(
    client: OpenAI,
    question: str,
    model: str = "kimi-vl-a3b-instruct",
    image_paths: list[str] | None = None,
    use_blind_prompt: bool = True,
    max_retries: int = 3,
) -> dict:
    """
    Send a single query to the Kimi API.
    Returns dict with 'response', 'tokens_used', 'latency_ms'.
    """
    prompt_template = BLIND_PROMPT if use_blind_prompt else GENERIC_PROMPT
    prompt_text = prompt_template.format(question=question)

    # Build message content
    content = []
    if image_paths:
        for img_path in image_paths:
            if os.path.exists(img_path):
                img_b64 = encode_image(img_path)
                ext = Path(img_path).suffix.lower().strip(".")
                mime = f"image/{ext}" if ext in ("png", "gif", "webp") else "image/jpeg"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{img_b64}"},
                })
    content.append({"type": "text", "text": prompt_text})

    messages = [{"role": "user", "content": content}]

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=256,
                temperature=0.2,  # low temp for evaluation consistency
            )
            elapsed_ms = (time.time() - start) * 1000

            result = response.choices[0].message.content.strip()
            tokens = response.usage.total_tokens if response.usage else 0

            return {
                "response": result,
                "tokens_used": tokens,
                "latency_ms": round(elapsed_ms, 1),
            }

        except Exception as e:
            print(f"  API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff
            else:
                return {
                    "response": "API_ERROR",
                    "tokens_used": 0,
                    "latency_ms": 0,
                }


# ============================================================
# Evaluation helpers
# ============================================================

def compute_token_overlap_f1(pred: str, ref: str) -> float:
    """Simple token F1 as a quick accuracy proxy."""
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = pred_tokens & ref_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_prediction(pred: str, references: list[str]) -> dict:
    """Score a prediction against reference answers."""
    best_f1 = max(compute_token_overlap_f1(pred, ref) for ref in references) if references else 0.0
    return {
        "max_f1": round(best_f1, 4),
        "pred_length_tokens": len(pred.split()),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run Kimi API baseline on EgoBlind test set")
    parser.add_argument("--test_data", required=True, help="EgoBlind test set JSON")
    parser.add_argument("--output_path", default="output/baseline_predictions.json")
    parser.add_argument("--api_key", default=None, help="Moonshot API key (or set MOONSHOT_API_KEY env)")
    parser.add_argument("--model", default="kimi-vl-a3b-instruct", help="Kimi model name")
    parser.add_argument("--frames_dir", default=None, help="Dir with extracted frames (optional)")
    parser.add_argument("--max_examples", type=int, default=None, help="Limit examples (for cost control)")
    parser.add_argument("--use_blind_prompt", action="store_true", default=True)
    parser.add_argument("--use_generic_prompt", action="store_true")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        raise ValueError("Provide --api_key or set MOONSHOT_API_KEY env variable")

    use_blind = not args.use_generic_prompt

    # Load test data
    with open(args.test_data, "r") as f:
        test_data = json.load(f)

    if args.max_examples:
        test_data = test_data[:args.max_examples]

    print(f"Running baseline on {len(test_data)} examples")
    print(f"Model: {args.model}")
    print(f"Prompt: {'blind-specific' if use_blind else 'generic'}")

    client = create_client(api_key)
    os.makedirs(Path(args.output_path).parent, exist_ok=True)

    results = []
    total_tokens = 0
    total_latency = 0

    for i, item in enumerate(test_data):
        question = item["question"]
        references = item.get("answers", [])
        q_type = item.get("question_type", item.get("type", ""))
        q_id = item.get("question_id", i)

        # Load frames if available
        image_paths = None
        if args.frames_dir:
            video_id = item.get("video_id", "")
            frame_dir = os.path.join(args.frames_dir, video_id)
            if os.path.isdir(frame_dir):
                image_paths = sorted(
                    [os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")]
                )[:8]  # max 8 frames

        # Query API
        api_result = query_kimi(
            client, question,
            model=args.model,
            image_paths=image_paths,
            use_blind_prompt=use_blind,
        )

        # Evaluate
        eval_result = evaluate_prediction(api_result["response"], references)

        results.append({
            "question_id": q_id,
            "question": question,
            "question_type": q_type,
            "prediction": api_result["response"],
            "references": references,
            "max_f1": eval_result["max_f1"],
            "pred_length": eval_result["pred_length_tokens"],
            "latency_ms": api_result["latency_ms"],
            "tokens_used": api_result["tokens_used"],
        })

        total_tokens += api_result["tokens_used"]
        total_latency += api_result["latency_ms"]

        if (i + 1) % 25 == 0:
            avg_f1 = sum(r["max_f1"] for r in results) / len(results)
            print(f"  [{i+1}/{len(test_data)}] avg_f1={avg_f1:.3f} tokens_so_far={total_tokens}")

        # Rate limiting — be gentle with the API
        time.sleep(0.5)

    # Save results
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Summary
    avg_f1 = sum(r["max_f1"] for r in results) / len(results)
    avg_latency = total_latency / len(results)
    avg_length = sum(r["pred_length"] for r in results) / len(results)

    # Per-type breakdown
    type_scores = {}
    for r in results:
        t = r["question_type"] or "unknown"
        if t not in type_scores:
            type_scores[t] = []
        type_scores[t].append(r["max_f1"])

    print("\n" + "=" * 60)
    print(f"BASELINE RESULTS ({len(results)} examples)")
    print(f"=" * 60)
    print(f"Overall F1:        {avg_f1:.4f}")
    print(f"Avg latency:       {avg_latency:.0f} ms")
    print(f"Avg response len:  {avg_length:.1f} tokens")
    print(f"Total tokens used: {total_tokens}")
    print(f"\nPer-type breakdown:")
    for t, scores in sorted(type_scores.items()):
        print(f"  {t:25s}  F1={sum(scores)/len(scores):.4f}  (n={len(scores)})")
    print(f"=" * 60)
    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
