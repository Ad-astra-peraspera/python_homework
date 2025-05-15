import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和负号正常显示
plt.rcParams['font.family'] = 'Songti SC'
import matplotlib.font_manager as fm
for font in fm.fontManager.ttflist:
    if 'Hei' in font.name or 'Ping' in font.name or 'Song' in font.name:
        print(font.name)
plt.rcParams['axes.unicode_minus'] = False

# ✅ 修改这里为你的实际路径
df = pd.read_excel("grade.xlsx")

# 需要统计的评分字段（1~5分）
likert_cols = ['内容', '结构', '语言', '交互']

# 构建二维数组：每个字段中1~5的计数
likert_data = []
for col in likert_cols:
    counts = df[col].value_counts().sort_index()
    row = [counts.get(i, 0) for i in range(1, 6)]
    likert_data.append(row)

likert_data = np.array(likert_data)

# 标签和颜色（与 Likert 等级对应）
labels = ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree']
colors = ['#d73027', '#fdae61', '#ffffbf', '#a6d96a', '#1a9850']

# 可视化：水平堆叠条形图
fig, ax = plt.subplots(figsize=(10, 6))
left = np.zeros(len(likert_cols))

for i in range(len(labels)):
    ax.barh(likert_cols, likert_data[:, i], left=left, color=colors[i], label=labels[i])
    for j in range(len(likert_cols)):
        if likert_data[j, i] > 0:
            ax.text(left[j] + likert_data[j, i] / 2, j, str(likert_data[j, i]),
                    ha='center', va='center', color='black')
    left += likert_data[:, i]

# 标题和图例设置
ax.set_title('评分分布（Likert 风格）')
ax.legend(loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.1))
ax.set_xlim(0, np.sum(likert_data, axis=1).max() + 10)
ax.invert_yaxis()
plt.tight_layout()
plt.show()