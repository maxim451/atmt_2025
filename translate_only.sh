#!/usr/bin/bash -l
#SBATCH --partition=teaching
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --gpus=2
#SBATCH --output=translate_only_%j.out

# --- modules & env ---
module load gpu
module load mamba
source activate atmt
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# --- paths (edit if your layout differs) ---
WORKDIR="$HOME/data/atmt_2025"        # your project root on cluster
cd "$WORKDIR"

SRC_TOKENIZER="cz-en/tokenizers/cz-bpe-8000.model"
TGT_TOKENIZER="cz-en/tokenizers/en-bpe-8000.model"
CKPT="cz-en/checkpoints/checkpoint_best.pt"

# Pick ONE dataset to evaluate (uncomment the block you want)

# --- Europarl (you made two plain-text files) ---
INPUT="europarl_2500.cs"              # source (Czech)
REF="europarl_2500.en"                 # reference (English)
OUT="cz-en/outputs/europarl_2500.hyp"  # hypotheses will be written here

## --- News Commentary (if you kept TSV with cs \t en) ---
## If your file is TSV, auto-split to temp src/ref once per run:
# TSV="news-commentary_2500.tsv"
# mkdir -p cz-en/outputs
# cut -f1 "$TSV" > cz-en/outputs/newscomm_2500.cs
# cut -f2 "$TSV" > cz-en/outputs/newscomm_2500.en
# INPUT="cz-en/outputs/newscomm_2500.cs"
# REF="cz-en/outputs/newscomm_2500.en"
# OUT="cz-en/outputs/newscomm_2500.hyp"

mkdir -p "$(dirname "$OUT")"

python translate.py \
  --cuda \
  --input "$INPUT" \
  --src-tokenizer "$SRC_TOKENIZER" \
  --tgt-tokenizer "$TGT_TOKENIZER" \
  --checkpoint-path "$CKPT" \
  --output "$OUT" \
  --max-len 300 \
  --bleu \
  --reference "$REF"