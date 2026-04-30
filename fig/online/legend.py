import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_COLOR_PULLING = "#7fcdbb"
BASE_COLOR_EXTRA = "#edf8b1"
LIGHTD_COLOR_PULLING = "#d9ebd4"
LIGHTD_COLOR_EXTRA = "#f8ac8c"

HATCH_1 = "||"
HATCH_2 = "--"
HATCH_3 = "\\"
HATCH_4 = "/"

# 创建一个空白图
fig, ax = plt.subplots(1, 1, figsize=(11, 1.0))

legend_elements = [
    Patch(
        facecolor=LIGHTD_COLOR_EXTRA,
        edgecolor="black",
        hatch=HATCH_4,
        label="BTS",
    ),
    Patch(
        facecolor=LIGHTD_COLOR_PULLING,
        edgecolor="black",
        hatch=HATCH_3,
        label="P2P",
    ),
    Patch(
        facecolor=BASE_COLOR_EXTRA,
        edgecolor="black",
        hatch=HATCH_1,
        label="LinUCB",
    ),
    Patch(
        facecolor=BASE_COLOR_PULLING,
        edgecolor="black",
        hatch=HATCH_2,
        label="epsGreedy",
    ),
]

# 创建图例
plt.legend(handles=legend_elements, ncol=4, mode="expand", prop={"size": 16})

# 关闭图的坐标轴
ax.axis("off")
plt.tight_layout()

# 保存图例到文件
plt.savefig("legend.pdf")

plt.show()
