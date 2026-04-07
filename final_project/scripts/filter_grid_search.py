import json

with open("results/all_results.json", "r") as f:
    results = json.load(f)

valid = [r for r in results if abs(r["params"]["alpha"] + r["params"]["beta"] + r["params"]["gamma"] - 1.0) < 1e-6]

print(f"Total configs: {len(results)}")
print(f"Configs with a+b+g=1: {len(valid)}")

if not valid:
    print("No valid configs found!")
else:
    best = min(valid, key=lambda r: r["overall"]["avg_loss"])
    print(f"\nBest config (by overall avg_loss):")
    print(json.dumps({"params": best["params"], "overall": best["overall"]}, indent=2))

    top5 = sorted(valid, key=lambda r: r["overall"]["avg_loss"])[:5]
    print(f"\nTop 5:")
    for i, r in enumerate(top5):
        p = r["params"]
        print(f"  {i+1}. loss={r['overall']['avg_loss']:.4f}  a={p['alpha']} b={p['beta']} g={p['gamma']} "
              f"tau_min={p['tau_min']} tau_max={p['tau_max']} p={p['p']} k={p['kappa']}")
