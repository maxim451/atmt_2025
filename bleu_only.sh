#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=0:10:0
#SBATCH --ntasks=1
#SBATCH --mem=2GB
#SBATCH --cpus-per-task=1
#SBATCH --output=out_bleu_compare.out

# === ENVIRONMENT SETUP ===
module load mamba
source activate atmt

echo "[INFO] Starting BLEU comparison..."
date

# Проверим, что sacrebleu установлен
python -m pip install --quiet sacrebleu

python - <<'PYCODE'
import sacrebleu, os

# === PATHS ===
ref_path = "cz-en/data/prepared/test_text.en"

variants = {
    "Baseline": "cz-en/outputs/test_output_baseline_trimmed.txt",
    "MQA": "cz-en/outputs/test_output_mqa_trimmed.txt",
    "Beam Search": "cz-en/outputs/test_output_mqa_beam5_trimmed.txt",
}

# === LOAD REFERENCE ===
with open(ref_path, encoding="utf-8") as f:
    refs = [line.strip() for line in f if line.strip()]

print(f"[INFO] Reference lines: {len(refs)}")

# === LOOP THROUGH VARIANTS ===
for name, hyp_path in variants.items():
    if not os.path.exists(hyp_path):
        print(f"[WARN] {name} output not found at {hyp_path}, skipping.")
        continue
    with open(hyp_path, encoding="utf-8") as f:
        hyps = [line.strip() for line in f if line.strip()]

    n = min(len(refs), len(hyps))
    refs_, hyps_ = refs[:n], hyps[:n]
    bleu = sacrebleu.corpus_bleu(hyps_, [refs_])
    print(f"{name:12s} | Lines: {n:5d} | BLEU: {bleu.score:6.3f}")

print("\n[INFO] BLEU comparison completed.")
PYCODE

date
