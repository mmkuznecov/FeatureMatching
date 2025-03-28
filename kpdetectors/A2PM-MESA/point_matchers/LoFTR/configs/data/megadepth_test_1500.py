'''
Author: Easonyesheng preacher@sjtu.edu.cn
Date: 2024-07-27 10:35:30
LastEditors: Easonyesheng preacher@sjtu.edu.cn
LastEditTime: 2024-07-27 14:52:28
FilePath: /A2PM-MESA/point_matchers/LoFTR/configs/data/megadepth_test_1500.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from point_matchers.LoFTR.configs.data.base import cfg

TEST_BASE_PATH = "assets/megadepth_test_1500_scene_info"

cfg.DATASET.TEST_DATA_SOURCE = "MegaDepth"
cfg.DATASET.TEST_DATA_ROOT = "data/megadepth/test"
cfg.DATASET.TEST_NPZ_ROOT = f"{TEST_BASE_PATH}"
cfg.DATASET.TEST_LIST_PATH = f"{TEST_BASE_PATH}/megadepth_test_1500.txt"

cfg.DATASET.MGDPT_IMG_RESIZE = 840
cfg.DATASET.MIN_OVERLAP_SCORE_TEST = 0.0
