# CAA

*Change-Aware Adapter Driving Vision Foundation Models for Remote Sensing Visual Change Understanding* (under review at IEEE TGRS)

## Checkpoints

`ckpts/cc.pth`, `ckpts/cd.pth`: https://pan.baidu.com/s/1nRiByotSk2PsiY3KVkA5KA?pwd=uwst (code: uwst)

## Training

```bash
CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch \
  --master_port=3153 --nproc_per_node=2 train_cd.py \
  --dataset levir --exp_dir osnet --model_name dinov3caax3hr_small \
  --rank 24 --depth 4 \
  --batch_size 16 --eval_batch_size 64 --epoch 100 --num_workers 4
```

```bash
CUDA_VISIBLE_DEVICES=3,4 python -m torch.distributed.launch \
  --master_port=3151 --nproc_per_node=2 train_cc.py \
  --dataset levir --exp_dir osnet --backbone dinov3_small \
  --n_enc 1 --n_dec 1 --dim 512 --max_len 41 \
  --rank 24 \
  --batch_size 20 --eval_batch_size 40 --epoch 10 --num_workers 4
```

## Inference

```bash
python eval_cd.py \
  --dataset levir --exp_dir osnet --exp_name bs16_small_r24 \
  --model_name dinov3caax3hr_small --rank 24 --depth 4 \
  --split TEST --batch_size 64
```

```bash
python eval_cc.py \
  --dataset levir --exp_dir osnet --exp_name bs20_small_r24 \
  --backbone dinov3_small --n_enc 1 --n_dec 1 --dim 512 --max_len 41 \
  --rank 24 --split TEST --beam_size 3 --batch_size 40
```

## Dataset

```
{data_folder}
├── RSICD
│   └── LEVIR-CD256
│       ├── A/
│       ├── B/
│       ├── label/
│       └── list/
│           ├── train.txt
│           ├── val.txt
│           └── test.txt
└── RSICC
    ├── Levir-CC-dataset
    │   └── images
    │       ├── train/{A,B}/
    │       ├── val/{A,B}/
    │       └── test/{A,B}/
    └── annotations
        ├── WORDMAP_LEVIR_CC_5_cap_per_img_5_min_word_freq.json
        ├── train_CAPTIONS_LEVIR_CC_5_cap_per_img_5_min_word_freq.json
        ├── val_CAPTIONS_LEVIR_CC_5_cap_per_img_5_min_word_freq.json
        ├── test_CAPTIONS_LEVIR_CC_5_cap_per_img_5_min_word_freq.json
        ├── train_ids.json
        ├── val_ids.json
        └── test_ids.json
```
