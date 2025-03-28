'''
Author: EasonZhang
Date: 2024-07-26 16:06:23
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2024-07-27 11:32:40
FilePath: /A2PM-MESA/point_matchers/ASpanFormer/configs/aspan/outdoor/aspan_test.py
Description: path update

Copyright (c) 2024 by EasonZhang, All Rights Reserved. 
'''
from point_matchers.ASpanFormer.src.config.default import _CN as cfg

cfg.ASPAN.COARSE.COARSEST_LEVEL= [36,36]
cfg.ASPAN.COARSE.TRAIN_RES = [832,832]
cfg.ASPAN.COARSE.TEST_RES = [1152,1152]
cfg.ASPAN.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.ASPAN.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3