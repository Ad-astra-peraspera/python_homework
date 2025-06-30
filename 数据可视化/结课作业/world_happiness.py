# Author: moqiHe
# Date: 2025-06-09
# Description:

from pyecharts.commons.utils import JsCode

import pandas as pd
from pyecharts.charts import Map, Bar, Pie, Page, Grid
from pyecharts import options as opts

# 读取数据
df = pd.read_csv('./happiness_2015_2019_avg.csv')

# 地图图表
map_data = list(zip(df["Country or region"], df["Score"]))
world_map = (
    Map(init_opts=opts.InitOpts(width="1000px", height="490px"))
    .add("幸福指数", map_data, "world", is_map_symbol_show=False, label_opts=opts.LabelOpts(is_show=False))
    .set_global_opts(
        title_opts=opts.TitleOpts(title="全球幸福指数地图"),
        visualmap_opts=opts.VisualMapOpts(max_=8, min_=2.5, is_piecewise=False, range_color=["#fef7f7","#fde8e6","#fbb4ae"] ),
        legend_opts=opts.LegendOpts(is_show=False)
    )
)

# Top 5 和 Bottom 5 数据
df_top5 = df.sort_values(by="Score", ascending=False).head(5)
df_bottom5 = df.sort_values(by="Score", ascending=True).head(5)

# Top 5 Bar
bar_top = (
    Bar(init_opts=opts.InitOpts(width="400px", height="230px"))
    .add_xaxis(df_top5["Country or region"].tolist())
    .add_yaxis("幸福指数", df_top5["Score"].tolist())
    .reversal_axis()
    .set_colors(["#decae5"])
    .set_global_opts(
        title_opts=opts.TitleOpts(title="幸福指数 Top 5 国家"),
        xaxis_opts=opts.AxisOpts(name="幸福指数", min_=2.5, max_=8),
        yaxis_opts=opts.AxisOpts(name="国家"),
        legend_opts=opts.LegendOpts(is_show=False)
    )
)
grid_top = (
    Grid(init_opts=opts.InitOpts(width="400px", height="240px"))
    .add(bar_top, grid_opts=opts.GridOpts(pos_right="20%",pos_left="40%"))  # 右边距防止标签遮挡
)

# Bottom 5 Bar
bar_bottom = (
    Bar(init_opts=opts.InitOpts(width="400px", height="240px"))
    .add_xaxis(df_bottom5["Country or region"].tolist())
    .add_yaxis("幸福指数", df_bottom5["Score"].tolist())
    .reversal_axis()
    .set_colors(["#b3cde4"])
    .set_global_opts(
        title_opts=opts.TitleOpts(title="幸福指数 Bottom 5 国家"),
        xaxis_opts=opts.AxisOpts(name="幸福指数", min_=2.5, max_=8),
        yaxis_opts=opts.AxisOpts(name="国家"),
        legend_opts=opts.LegendOpts(is_show=False)
    )
)
grid_bottom = (
    Grid(init_opts=opts.InitOpts(width="400px", height="240px"))
    .add(bar_bottom, grid_opts=opts.GridOpts(pos_right="20%",pos_left="40%"))
)

# 饼图
avg_factors = df[[
    "GDP per capita", "Social support", "Healthy life expectancy",
    "Freedom to make life choices", "Generosity", "Perceptions of corruption"
]].mean()
pie_data = [(k, round(v, 3)) for k, v in avg_factors.items()]
pie = (
    Pie(init_opts=opts.InitOpts(width="520px", height="320px"))
    .add("", pie_data, label_opts=opts.LabelOpts(formatter="{b}: {d}%"),
        itemstyle_opts=opts.ItemStyleOpts(border_color="black", border_width=0.5))
    .set_colors(["#fbb4ae", "#ffd9a8", "#b3cde4", "#cbe9c3", "#decae5","#f8eaac"])
    .set_global_opts(
        title_opts=opts.TitleOpts(title="幸福感贡献因子占比（均值）"),
        legend_opts=opts.LegendOpts(is_show=False)
    )
    .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {d}%"))
)
from pyecharts.charts import Line
from pyecharts import options as opts

# 准备数据
df_top10 = df.sort_values(by="Score", ascending=False).head(10)
df_bottom10 = df.sort_values(by="Score", ascending=True).head(10)

keys = ["GDP per capita", "Social support", "Healthy life expectancy",
        "Freedom to make life choices", "Generosity", "Perceptions of corruption"]
x_labels = ["GDP", "Social support", "Healthy life", "Freedom", "Generosity", "Corruption"]

# 平均值列表（保留两位小数）
top_values = [round(df_top10[k].mean(), 4) for k in keys]
bottom_values = [round(df_bottom10[k].mean(), 4) for k in keys]

# 折线图
line = (
    Line(init_opts=opts.InitOpts(width="900px", height="320px"))
    .add_xaxis(x_labels)
    .add_yaxis(
        "Top 10 国家均值",
        top_values,
        itemstyle_opts=opts.ItemStyleOpts(color="#fbb4ae"),
        label_opts=opts.LabelOpts(formatter="{c}")  # 默认保留原始值，建议你控制数据精度而不是显示精度
    )
    .add_yaxis(
        "Bottom 10 国家均值",
        bottom_values,
        itemstyle_opts=opts.ItemStyleOpts(color="#b3cde4"),
        label_opts=opts.LabelOpts(formatter="{c}")
    )
    .set_global_opts(
        title_opts=opts.TitleOpts(title="幸福因子均值对比"),
        legend_opts=opts.LegendOpts(pos_top="5%"),
        yaxis_opts=opts.AxisOpts(min_=0)  # Y轴从0开始更美观
    )
)

# 创建可拖拽仪表盘页面
page = Page(layout=Page.DraggablePageLayout)
page.add(world_map, grid_top, grid_bottom, pie, line)
page.render("happiness_dashboard_1.html")