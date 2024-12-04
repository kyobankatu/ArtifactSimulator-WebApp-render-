from flask import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pyocr
import re

# 率、ダメ
CRIT = np.array([54, 62, 70, 78])
# 攻撃
ATK = np.array([41, 47, 53, 58])
# スコア確率
NUMS = np.array([41, 47, 53, 58, 54, 62, 70, 78, 54, 62, 70, 78, 0, 0, 0, 0])
# 初期オプ数
OPTION = 4
# 初期オプに含まれているか
IS_CRIT_DMG = True
IS_CRIT_RATE = True
IS_ATK = True
# 初期スコア
INIT_SCORE = 0
# 調査スコア
SCORE = 0
# 強化回数
COUNT = 5
# GUIフォント
FONT_TYPE = "meiryo"

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def hello_world():
    return 'Hello World!'

class Calculator():
    # スコアの伸びの分布を計算 (indexが伸び幅の)
    def getDistribution(self, nums, count):
        dp = np.zeros((count + 1, max(nums) * count + 1))
        dp[0][0] = 1.0

        for i in range(count):
            for num in nums:
                prev = dp[i][:dp.shape[1] - num]
                dp[i + 1][num:] += prev / nums.shape[0]
        
        return dp[count]

    def calculate(self):
        if OPTION == 4:
            y = self.getDistribution(NUMS, COUNT)
            return y
        else:
            NUMS_4OP = []
            if not IS_CRIT_DMG:
                tmp = np.copy(NUMS)
                tmp[12:] = CRIT
                NUMS_4OP.append(tmp)
            if not IS_CRIT_RATE:
                tmp = np.copy(NUMS)
                tmp[12:] = CRIT
                NUMS_4OP.append(tmp)
            if not IS_ATK:
                tmp = np.copy(NUMS)
                tmp[12:] = ATK
                NUMS_4OP.append(tmp)
            
            main_probability = (7 - len(NUMS_4OP)) / 7
            sub_probability = 0
            if len(NUMS_4OP) != 0:
                sub_probability = (1 - main_probability) / len(NUMS_4OP)

            y = np.zeros(np.amax(CRIT) * COUNT + 1) # COUNT = 5

            main_y = self.getDistribution(NUMS, 4)
            y[0:main_y.shape[0]] += main_y * main_probability

            for nums in NUMS_4OP: 
                for num_4th in nums:
                    sub_y = self.getDistribution(nums, 4)
                    y[num_4th:num_4th + sub_y.shape[0]] += sub_y / len(nums) * sub_probability

            return y

if __name__ == "__main__":
    app.run()
