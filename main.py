from flask import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pyocr
import re
import io

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

@app.route("/get-data", methods=["POST"])
def get_data():
    global NUMS, OPTION, IS_CRIT_DMG, IS_CRIT_RATE, IS_ATK, INIT_SCORE, SCORE, COUNT

    # リクエストから数値を取得
    data = request.get_json()

    OPTION = int(data['option'])
    IS_CRIT_DMG = bool(data['crit_dmg'])
    IS_CRIT_RATE = bool(data['crit_rate'])
    IS_ATK = bool(data['atk'])
    INIT_SCORE = int(data['init'])
    SCORE = int(data['score'])
    COUNT = int(data['count'])

    NUMS = np.array([41, 47, 53, 58, 54, 62, 70, 78, 54, 62, 70, 78, 0, 0, 0, 0])
    if not IS_ATK:
        NUMS[0:4] = 0
    if not IS_CRIT_DMG:
        NUMS[4:8] = 0
    if not IS_CRIT_RATE:
        NUMS[8:12] = 0

    calc = Calculator()
    y = calc.calculate()
    x = np.zeros(y.shape[0])
    for i in range(x.shape[0]):
        x[i] = i / 10.0

    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(INIT_SCORE + x, y, width = 0.05)

    # グラフをメモリ内の画像として保存
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

class Calculator():
    # スコアの伸びの分布を計算 (indexが伸び幅の10倍整数)
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

            y = np.zeros(np.amax(CRIT) * COUNT + 1)

            main_y = self.getDistribution(NUMS, COUNT - 1)
            y[0:main_y.shape[0]] += main_y * main_probability

            for nums in NUMS_4OP:
                for num_4th in nums:
                    sub_y = self.getDistribution(nums, COUNT - 1)
                    y[num_4th:num_4th + sub_y.shape[0]] += sub_y / len(nums) * sub_probability

            return y

if __name__ == "__main__":
    app.run()
