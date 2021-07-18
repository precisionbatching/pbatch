#!/bin/sh

vocab="data/data/vocab.json"
train_src="data/data/train.de-en.de.wmixerprep"
train_tgt="data/data/train.de-en.en.wmixerprep"
dev_src="data/data/valid.de-en.de"
dev_tgt="data/data/valid.de-en.en"
test_src="data/data/test.de-en.de"
test_tgt="data/data/test.de-en.en"

work_dir="work_dir"

mkdir -p ${work_dir}
echo save results to ${work_dir}

#python nmt.py \
#    train \
#    --cuda \
#    --vocab ${vocab} \
#    --train-src ${train_src} \
#    --train-tgt ${train_tgt} \
#    --dev-src ${dev_src} \
#    --dev-tgt ${dev_tgt} \
#    --input-feed \
#    --valid-niter 2400 \
#    --batch-size 64 \
#    --hidden-size 1024 \
#    --embed-size 1024 \
#    --uniform-init 0.1 \
#    --label-smoothing 0.1 \
#    --dropout 0.2 \
#    --clip-grad 5.0 \
#    --save-to ${work_dir}/model.bin \
#    --lr-decay 0.5 2>${work_dir}/err.log#

python nmt.py \
    decode \
    --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt