# Revisiting Token Dropping Strategy in Efficient BERT Pretraining

This repository contains the code for our paper submitted to ACL2023.

## Requirements and Installation

- PyTorch version >= 1.10.0
- Python version >= 3.8
- For training, you'll also need an NVIDIA GPU and NCCL.
- To install **fairseq** and develop locally:

``` bash
git clone https://github.com/facebookresearch/fairseq.git
mv fairseq fairseq-setup
cd fairseq-setup
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./
```

# Getting Started
Here, we introduce how to perform our **ScTD** strategy.

# ScTD for BERT pretraining
To perform this process, you should first prepare the training environment by the following commands:

``` 
# removing the original scripts
rm -r fairseq-setup/fairseq
rm -r fairseq-setup/fairseq_cli/train.py

# using our self-questioning scripts
cp -r fairseq-sctd fairseq-setup/
mv fairseq-setup/fairseq-sctd fairseq-setup/fairseq
```

Then, you can start ScTD training by the following commands:

``` 
DATA_DIR=data-path
SAVE_DIR=save-path

mkdir -p  $SAVE_DIR

fairseq-train $DATA_DIR \
    --train-subset train \
    --valid-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --memory-efficient-fp16 \
    --fp16-init-scale 8 \
    --arch roberta_large \
    --task masked_lm \
    --sample-break-mode "complete" \
    --batch-size 8 \
    --tokens-per-sample 512 \
    --save-interval 1 --save-interval-updates 10000 --keep-interval-updates 25 \
    --no-epoch-checkpoints \
    --optimizer adam --adam-betas "(0.9, 0.999)" \
    --adam-eps 1e-06 \
    --weight-decay 0.01 \
    --clip-norm 0.0 \
    --lr 2e-4 \
    --lr-scheduler polynomial_decay \
    --warmup-updates 25000 \
    --dropout 0.1 \
    --token-drop 0.5 \
    --max-positions 512 \
    --attention-dropout 0.1 \
    --update-freq 2 \
    --ddp-backend=legacy_ddp \
    --total-num-update 250000 \
    --max-update 250000 \
    --max-epoch 40 \
    --save-dir $SAVE_DIR \
    --log-format json --log-interval 100 2>&1 | tee $SAVE_DIR/train.log

```

# Evaluation
You can evaluate the pretrained models by using the original [fine-tuning scripts](https://github.com/facebookresearch/fairseq/tree/main/examples/roberta), or any way you like.



