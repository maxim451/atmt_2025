#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=2:0:0
#SBATCH --ntasks=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=5
#SBATCH --gpus=1
#SBATCH --output=out_translate_test.out

# === ENVIRONMENT SETUP ===
module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

echo "[INFO] Starting translation on test set..."
date

# === STEP 1: Convert pickled datasets to text if needed ===
echo "[INFO] Checking format of test files..."
python - <<'PYCODE'
import pickle, os, sys

def convert_if_pickle(in_path, out_path):
    try:
        with open(in_path, "rb") as f:
            data = pickle.load(f)
        # if load succeeded — write plain text
        with open(out_path, "w", encoding="utf-8") as out:
            for seq in data:
                if isinstance(seq, (list, tuple)):
                    out.write(" ".join(map(str, seq)) + "\n")
                else:
                    out.write(str(seq) + "\n")
        print(f"[OK] Converted {in_path} -> {out_path} ({len(data)} lines)")
    except Exception as e:
        print(f"[SKIP] {in_path} is already text ({e})")

convert_if_pickle("cz-en/data/prepared/test.cz", "cz-en/data/prepared/test_text.cz")
convert_if_pickle("cz-en/data/prepared/test.en", "cz-en/data/prepared/test_text.en")
PYCODE

# === STEP 2: TRANSLATE ===
python translate.py \
    --cuda \
    --input cz-en/data/prepared/test_text.cz \
    --reference cz-en/data/prepared/test_text.en \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output cz-en/outputs/test_output.txt \
    --max-len 300 \
    --bleu

echo "[INFO] Translation completed."
date
