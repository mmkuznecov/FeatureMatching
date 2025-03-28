'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2024-07-27 10:35:30
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2024-07-27 14:51:26
FilePath: /A2PM-MESA/point_matchers/LoFTR/configs/loftr/indoor/scannet/loftr_ds_eval_new.py
Description: change path
'''
""" A config only for reproducing the ScanNet evaluation results.

We remove border matches by default, but the originally implemented
`remove_border()` has a bug, leading to only two sides of
all borders are actually removed. However, the [bug fix](https://github.com/zju3dv/LoFTR/commit/e9146c8144dea5f3cbdd98b225f3e147a171c216)
makes the scannet evaluation results worse (auc@10=40.8 => 39.5), which should be
caused by tiny result fluctuation of few image pairs. This config set `BORDER_RM` to 0
to be consistent with the results in our paper.

Update: This config is for testing the re-trained model with the pos-enc bug fixed.
"""

from point_matchers.LoFTR.src.config.default import _CN as cfg

cfg.LOFTR.COARSE.TEMP_BUG_FIX = True
cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'

cfg.LOFTR.MATCH_COARSE.BORDER_RM = 0
