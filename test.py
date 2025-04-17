import subprocess
import time
import os

net_ids = ["mcunet-in0", "mcunet-in1", "mcunet-in2", "mcunet-in3", "mcunet-in4"]
data_dir = os.path.expanduser("~/datasets/imagenet/val")
results = []

print(f"{'Model':<12} {'Top-1':>7} {'Top-5':>7} {'Time(s)':>8}")
print("-" * 38)

for net_id in net_ids:
    start = time.time()
    # Run eval_torch.py as subprocess, capture stdout
    try:
        output = subprocess.check_output(
            ["python3", "eval_torch.py", "--net_id", net_id, "--dataset", "imagenet", "--data-dir", data_dir],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        elapsed = time.time() - start

        # Parse output
        top1 = top5 = "N/A"
        for line in output.splitlines():
            if "Top-1 Acc" in line:
                top1 = line.split(":")[-1].strip()
            elif "Top-5 Acc" in line:
                top5 = line.split(":")[-1].strip()

        results.append((net_id, top1, top5, f"{elapsed:.2f}"))
        print(f"{net_id:<12} {top1:>7} {top5:>7} {elapsed:>8.2f}")

    except subprocess.CalledProcessError as e:
        print(f"{net_id:<12}  [FAILED]")
        print(e.output)

# Optional: write results to CSV
with open("benchmark_results.csv", "w") as f:
    f.write("Model,Top-1,Top-5,Time(s)\n")
    for r in results:
        f.write(",".join(r) + "\n")