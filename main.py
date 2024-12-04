from flask import *
import numpy as np
import matplotlib.pyplot as plt
import io

# 定数
CRIT = np.array([54, 62, 70, 78])
ATK = np.array([41, 47, 53, 58])
NUMS_DEFAULT = np.array([41, 47, 53, 58, 54, 62, 70, 78, 54, 62, 70, 78, 0, 0, 0, 0])
FONT_TYPE = "meiryo"

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def hello_world():
    return 'Hello World!'

@app.route("/get-data", methods=["POST"])
def get_data():
    # リクエストから数値を取得
    data = request.get_json()

    option = int(data['option'])
    is_crit_dmg = bool(data['crit_dmg'])
    is_crit_rate = bool(data['crit_rate'])
    is_atk = bool(data['atk'])
    init_score = int(data['init'])
    score = int(data['score'])
    count = int(data['count'])

    # NUMSをリセット
    nums = np.copy(NUMS_DEFAULT)

    # オプションに応じてNUMSを調整
    if not is_atk:
        nums[0:4] = 0
    if not is_crit_dmg:
        nums[4:8] = 0
    if not is_crit_rate:
        nums[8:12] = 0

    calc = Calculator(option, is_crit_dmg, is_crit_rate, is_atk, nums, init_score, score, count)
    y = calc.calculate()
    x = np.zeros(y.shape[0])
    for i in range(x.shape[0]):
        x[i] = i / 10.0

    # グラフを作成
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(init_score + x, y, width=0.05)

    # グラフをメモリ内の画像として保存
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return send_file(img, mimetype='image/png')

class Calculator():
    def __init__(self, option, is_crit_dmg, is_crit_rate, is_atk, nums, init_score, score, count):
        self.option = option
        self.is_crit_dmg = is_crit_dmg
        self.is_crit_rate = is_crit_rate
        self.is_atk = is_atk
        self.nums = nums
        self.init_score = init_score
        self.score = score
        self.count = count

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
        if self.option == 4:
            y = self.getDistribution(self.nums, self.count)
            return y
        else:
            nums_4op = []
            if not self.is_crit_dmg:
                tmp = np.copy(self.nums)
                tmp[12:] = CRIT
                nums_4op.append(tmp)
            if not self.is_crit_rate:
                tmp = np.copy(self.nums)
                tmp[12:] = CRIT
                nums_4op.append(tmp)
            if not self.is_atk:
                tmp = np.copy(self.nums)
                tmp[12:] = ATK
                nums_4op.append(tmp)

            main_probability = (7 - len(nums_4op)) / 7
            sub_probability = 0
            if len(nums_4op) != 0:
                sub_probability = (1 - main_probability) / len(nums_4op)

            y = np.zeros(np.amax(CRIT) * self.count + 1)

            main_y = self.getDistribution(self.nums, self.count - 1)
            y[0:main_y.shape[0]] += main_y * main_probability

            for nums in nums_4op:
                for num_4th in nums:
                    sub_y = self.getDistribution(nums, self.count - 1)
                    y[num_4th:num_4th + sub_y.shape[0]] += sub_y / len(nums) * sub_probability

            return y

if __name__ == "__main__":
    app.run(threaded=True)
