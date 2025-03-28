'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2024-07-27 10:35:30
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2024-07-27 14:58:16
FilePath: /A2PM-MESA/point_matchers/LoFTR/configs/loftr/outdoor/buggy_pos_enc/loftr_ds.py
Description: path modification
'''
from point_matchers.LoFTR.src.config.default import _CN as cfg

cfg.LOFTR.COARSE.TEMP_BUG_FIX = False
cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3
