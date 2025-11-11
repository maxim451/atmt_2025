#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=2
#SBATCH --gpus=1
#SBATCH --output=out_mqa_test.out

module load gpu
module load mamba
source activate atmt

echo "[INFO] Quick test of Multi-Query Attention..."
date

python train.py \
    --cuda \
    --data cz-en/data/prepared/ \
    --src-tokenizer cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer cz-en/tokenizers/en-bpe-8000.model \
    --source-lang cz \
    --target-lang en \
    --arch transformer \
    --batch-size 8 \
    --max-epoch 1 \
    --ignore-checkpoints \
    --dim-embedding 128 \
    --attention-heads 4 \
    --dim-feedforward-encoder 256 \
    --dim-feedforward-decoder 256 \
    --max-seq-len 100 \
    --n-encoder-layers 2 \
    --n-decoder-layers 2 \
    --log-file logs/mqa_test.log

echo "[INFO] Finished quick test."
date
