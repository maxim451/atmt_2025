#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=4GB
#SBATCH --cpus-per-task=1
#SBATCH --output=fix_prepared_data.out

# Загрузим окружение
module load mamba
source activate atmt

echo "[INFO] Удаляем повреждённые файлы..."
rm -f cz-en/data/prepared/test.cz cz-en/data/prepared/test.en

echo "[INFO] Запускаем препроцессинг..."
python preprocess.py \
    --source-lang cz \
    --target-lang en \
    --raw-data ~/shares/cz-en/data/raw \
    --dest-dir ./cz-en/data/prepared \
    --model-dir ./cz-en/tokenizers \
    --test-prefix test \
    --train-prefix train \
    --valid-prefix valid \
    --src-vocab-size 8000 \
    --tgt-vocab-size 8000 \
    --src-model ./cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-model ./cz-en/tokenizers/en-bpe-8000.model

echo "[INFO] Проверяем файлы..."
file cz-en/data/prepared/test.cz
file cz-en/data/prepared/test.en

echo "[INFO] Размеры файлов:"
ls -lh cz-en/data/prepared/test.cz cz-en/data/prepared/test.en

echo "[DONE] Проверка завершена"
