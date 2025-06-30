# Author: moqiHe
# Date: 2025-06-04
# Description:
# -*- coding: utf-8 -*-
"""
脚本功能：从 Excel 文件中读取阿尔及利亚地区的设备安装记录，
           按省份统计安装台数，然后基于 GeoJSON 地图绘制省级热力图。
依赖库：pandas, geopandas, matplotlib
"""

import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


# 示例：如果你在 macOS 上，系统里常见的中文字体包括 "Heiti TC"、"STHeiti"
# 如果你在 Windows 上，可以改成 "SimHei"、"Microsoft YaHei" 等
plt.rcParams['font.sans-serif'] = ['Heiti TC']    # 或 ['SimHei']、['Microsoft YaHei'] 等
plt.rcParams['axes.unicode_minus'] = False


def load_and_clean_data(excel_path: str) -> pd.DataFrame:
    """
    功能：从 Excel 文件中读取原始安装记录，提取并清洗字段。
    输入：
        excel_path (str) - Excel 文件的相对或绝对路径
            示例： "Installations Maccura.xlsx"
    输出：
        pd.DataFrame - 包含 ['Product', 'Lab', 'City', 'Province', 'Date'] 列的清洗后 DataFrame
    主要步骤：
        1. 用 pandas.read_excel 读取整个工作表。
        2. 假定原始表格中第 2 行（索引 1）开始存放安装记录，前 7 列依次是：
            Unnamed:0 -> Product
            Unnamed:3 -> Lab
            Unnamed:4 -> City
            Unnamed:5 -> Province
            Unnamed:6 -> Date
        3. 丢弃不需要的中间列，只保留上述五个字段。
        4. 对 Province 列做 .strip().title() 处理，去除多余空格并统一为首字母大写其余小写。
    """
    # 1. 读取整个 Excel 表格
    raw = pd.read_excel(excel_path)

    # 2. 取从第二行开始的安装记录，前 7 列
    data = raw.iloc[1:, 0:7].copy()
    data.columns = ['Product', 'Serial1', 'Serial2', 'Lab', 'City', 'Province', 'Date']

    # 3. 只保留我们需要的五列
    data = data[['Product', 'Lab', 'City', 'Province', 'Date']]

    # 4. 清洗 Province 列：去除首尾空白并将每个单词首字母大写
    data['Province'] = data['Province'].astype(str).str.strip().str.title()

    return data


def aggregate_by_province(df: pd.DataFrame) -> pd.DataFrame:
    """
    功能：按省份统计安装（销售）数量。
    输入：
        df (pd.DataFrame) - 已清洗的安装记录 DataFrame，包含 'Province' 列
    输出：
        pd.DataFrame - 两列：['Province', 'Count']，代表各省的安装总数（Count）
    主要步骤：
        1. 使用 df['Province'].value_counts() 统计每个省出现的次数（安装台数）。
        2. 将结果重置索引并重命名为 ['Province', 'Count']。
    """
    counts = df['Province'].value_counts().reset_index()
    counts.columns = ['Province', 'Count']
    return counts


def plot_algeria_choropleth(shapefile_path: str, province_counts: pd.DataFrame):
    """
    功能：读取阿尔及利亚省级 Shapefile，并根据各省安装数绘制热力图。
    输入：
        shapefile_path (str)       - 阿尔及利亚省级 Shapefile 的路径，须指向 .shp 文件
                                     示例： "gadm41_DZA_shp/gadm41_DZA_1.shp"
        province_counts (DataFrame) - 包含 ['Province', 'Count'] 的 DataFrame
    主要步骤：
        1. 检查 Shapefile 是否存在，否则抛出 FileNotFoundError 提示。
        2. 用 geopandas.read_file 读取 .shp，得到 GeoDataFrame（几何和属性）。
        3. 假定 GADM Shapefile 中的省份名称字段为 'NAME_1'，先对它做 .strip().title() 统一格式。
        4. 将 province_counts 与 GeoDataFrame 合并：left_on='NAME_1', right_on='Province', how='left'。
        5. 对缺失的 Count 值填 0。
        6. 用 GeoDataFrame.plot(column='Count', cmap='OrRd', legend=True) 绘制热力图。
    """
    # 1. 检查 .shp 文件是否存在
    if not os.path.exists(shapefile_path):
        raise FileNotFoundError(f"找不到 Shapefile 文件：{shapefile_path}\n"
                                "请检查文件名和文件夹路径是否正确，且 .shp/.shx/.dbf/.prj 配套齐全。")

    # 2. 读取 Shapefile，得到 GeoDataFrame
    gdf = gpd.read_file(shapefile_path)

    # 可选：打印列名，确认要素中含有 'NAME_1'
    # print("Shapefile 属性列：", gdf.columns.tolist())

    # 3. 统一 GeoDataFrame 中 'NAME_1' 的格式：去空格 + 首字母大写
    gdf['NAME_1'] = gdf['NAME_1'].astype(str).str.strip().str.title()

    # 4. 合并 GeoDataFrame 与 province_counts
    merged = gdf.merge(province_counts, left_on='NAME_1', right_on='Province', how='left')

    # 5. 缺失的 Count 填 0
    merged['Count'] = merged['Count'].fillna(0)

    # 6. 绘制 Choropleth 热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    merged.plot(
        column='Count',           # 按哪个字段上色
        cmap='OrRd',              # 颜色映射：橙-红渐变
        linewidth=0.8,            # 省界线宽度
        edgecolor='0.8',          # 省界线颜色（灰度 0.8）
        legend=True,              # 显示图例
        ax=ax,
        missing_kwds={            # 对于缺省（NaN）的省份，用灰色+斜线表示“无数据”
            "color": "lightgrey",
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data"
        }
    )

    # 添加标题并去坐标轴刻度
    ax.set_title("阿尔及利亚各省销售热力图", fontsize=16)
    ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # —————— 一、文件路径配置 ——————
    # 1. Excel 原始安装数据（与 alg.py 同目录）
    excel_path = "Installations Maccura.xlsx"
    # 2. GADM Shapefile 目录里的 .shp 文件（相对路径）
    shapefile_path = "gadm41_DZA_shp/gadm41_DZA_1.shp"

    # —————— 二、读取并清洗安装数据 ——————
    print("正在读取并清洗 Excel 安装数据……")
    df_clean = load_and_clean_data(excel_path)
    # df_clean 示例：
    #     Product           Lab      City  Province        Date
    # 1   Maccura i1000  LABM DR...  Douera    Alger  2022-03-07
    # 2   Maccura i1000  LABM DR...  Alger    Alger  2022-03-28
    # …

    # —————— 三、按省份统计安装台数 ——————
    print("正在统计各省安装台数……")
    province_counts = aggregate_by_province(df_clean)
    # 打印前几行结果方便检查
    print("各省安装台数统计结果（示例前 10 行）：")
    print(province_counts.head(10))

    # —————— 四、绘制阿尔及利亚省级热力图 ——————
    print("正在读取 Shapefile 并绘制热力图……")
    plot_algeria_choropleth(shapefile_path, province_counts)