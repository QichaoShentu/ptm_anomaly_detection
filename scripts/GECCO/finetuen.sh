python -u finetune.py --save_name GECCOW100P1MMaskR320Baseline --data NIPS_TS_GECCO --win_size 100 --patch_len 1 --mask_mode M_binomial --repr_dims 320 --ratio 1 --finetune 0 --eval
python -u finetune.py --save_name GECCOW100P1MaskR320Baseline --data NIPS_TS_GECCO --win_size 100 --patch_len 1 --mask_mode binomial --repr_dims 320 --ratio 1 --finetune 0 --eval
python -u finetune.py --save_name GECCOW100P5MMaskR320Baseline --data NIPS_TS_GECCO --win_size 100 --patch_len 5 --mask_mode M_binomial --repr_dims 320 --ratio 1 --finetune 0 --eval
python -u finetune.py --save_name GECCOW100P5MaskR320Baseline --data NIPS_TS_GECCO --win_size 100 --patch_len 5 --mask_mode binomial --repr_dims 320 --ratio 1 --finetune 0 --eval
python -u finetune.py --save_name GECCOW100P1MMaskR320Finetune --data NIPS_TS_GECCO --win_size 100 --patch_len 1 --mask_mode M_binomial --repr_dims 320 --ptm /workspace/ptm_anomaly_detection/training1/W100P1MMaskR320-20231012/model.pkl --ratio 1 --finetune 1 --eval
python -u finetune.py --save_name GECCOW100P1MaskR320Finetune --data NIPS_TS_GECCO --win_size 100 --patch_len 1 --mask_mode binomial --repr_dims 320 --ptm /workspace/ptm_anomaly_detection/training1/W100P1MaskR320-20231012/model.pkl --ratio 1 --finetune 1 --eval
python -u finetune.py --save_name GECCOW100P5MMaskR320Finetune --data NIPS_TS_GECCO --win_size 100 --patch_len 5 --mask_mode M_binomial --repr_dims 320 --ptm /workspace/ptm_anomaly_detection/training1/W100P5MMaskR320-20231012/model.pkl --ratio 1 --finetune 1 --eval
python -u finetune.py --save_name GECCOW100P5MaskR320Finetune --data NIPS_TS_GECCO --win_size 100 --patch_len 5 --mask_mode binomial --repr_dims 320 --ptm /workspace/ptm_anomaly_detection/training1/GECCOW100P5MaskR320-20231012/model.pkl --ratio 1 --finetune 1 --eval

