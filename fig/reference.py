import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 色彩
BASE_COLOR_PULLING = "#7fcdbb"
BASE_COLOR_EXTRA = "#edf8b1"
LIGHTD_COLOR_PULLING = "#d9ebd4"
LIGHTD_COLOR_EXTRA = "#f8ac8c"

# datacenter和edge的花纹样式
HATCH_1 = "||"
HATCH_2 = "--"
HATCH_3 = "\\"
HATCH_4 = "/"

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# 控制hatch花纹线条宽度
plt.rcParams["hatch.linewidth"] = 0.3

x = ["1", "2", "3", "4"]

y1 = [32.323, 7.350, 7.1415, 3.933]
y2 = [31.811, 6.924, 6.7425, 3.561]
t1 = [y1[i] - y2[i] for i in range(len(y1))]

y3 = [32.1783, 8.571, 7.588, 3.55]
y4 = [31.8412, 7.2466, 6.546, 2.85]
t2 = [y3[i] - y4[i] for i in range(len(y1))]

# y1 = [3.21, 7.350, 7.1415, 3.933]
# y2 = [2.85, 6.924, 6.7425, 3.561]
# t1 = [y1[i] - y2[i] for i in range(len(y1))]
#
# y3 = [3.75, 8.571, 7.588, 3.55]
# y4 = [3.01, 7.2466, 6.546, 2.85] # 0.7 访问

# 创建画布，控制长宽比和大小
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
N = 4
ind = np.arange(N)
bar_width = 0.2  # 条形宽度

ax.bar(
    ind - bar_width / 2,
    height=t1,
    width=bar_width,
    color=BASE_COLOR_EXTRA,
    linewidth=0,
    edgecolor="black",
    label="base Extra time",
    hatch=HATCH_1,
)
ax.bar(
    ind - bar_width / 2,
    height=y2,
    width=bar_width,
    color=BASE_COLOR_PULLING,
    linewidth=0,
    edgecolor="black",
    label="base Pulling time",
    hatch=HATCH_3,
    bottom=t1,
)

ax.bar(
    ind + bar_width / 2,
    height=t2,
    width=bar_width,
    color=LIGHTD_COLOR_EXTRA,
    linewidth=0,
    edgecolor="black",
    label="lightD Extra time",
    hatch=HATCH_2,
)
ax.bar(
    ind + bar_width / 2,
    height=y4,
    width=bar_width,
    color=LIGHTD_COLOR_PULLING,
    linewidth=0,
    edgecolor="black",
    label="lightD Pulling time",
    hatch=HATCH_4,
    bottom=t2,
)

font2 = {
    "family": "Helvetica",
    "weight": "normal",
    "size": 35,
}

plt.xlabel("Number of iterations", font2)
plt.ylabel("Completion time (sec)", font2, loc="top")

ax.set_axisbelow(True)

# Add grid lines
ax.grid(axis="y", color="#A8BAC4", lw=1.2)

# Customize bottom spine
ax.spines["bottom"].set_lw(1.2)
ax.spines["bottom"].set_capstyle("butt")

plt.tick_params(labelsize=35)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname("Helvetica") for label in labels]

new_x = ["" + i for i in x]
plt.xticks(np.arange(N), new_x)

# 设置横纵坐标的名称以及对应字体格式

plt.tight_layout()
plt.savefig("500M.pdf", format="pdf")
plt.show()
