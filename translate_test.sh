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

# === TRANSLATE TEST SET ===
python translate.py \
    --cuda \
    --input cz-en/data/prepared/test.cz \
    --reference cz-en/data/prepared/test.en \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path cz-en/checkpoints/checkpoint_best.pt \
    --output cz-en/outputs/test_output.txt \
    --max-len 300 \
    --bleu

echo "[INFO] Translation completed."
date