import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Heiti TC']
plt.rcParams['axes.unicode_minus'] = False

# 血球仪数据
cbc_data = {
    'Alger': 4,
    'Tipaza': 3,
    'Skikda': 2,
    'Chlef': 2,
    'Guelma': 1,
    'Constantine': 1,
    'Relizane': 1,
    'Setif': 1,
    'Sidi Bel Abbes': 1
}

# 发光仪数据
clia_data = {
    'Alger': 21,
    'Oran': 10,
    'Mostaganem': 8,
    'Annaba': 6,
    'Batna': 5,
    'Biskra': 5,
    'Sétif': 4,
    'Aïn Defla': 4,
    'Mila': 4,
    'El Oued': 4,
    'Chlef': 4,
    'Mascara': 4,
    'Constantine': 4,
    "M'Sila": 3,
    'Tiaret': 3,
    'Médéa': 3,
    'Bordj Bou Arréridj': 2,
    'Ghardaïa': 2,
    'Tlemcen': 2,
    'Djelfa': 2,
    'Sidi Bel Abbès': 2,
    'Ouargla': 3,
    'Aïn Témouchent': 1,
    'Souk Ahras': 1,
    'Bouira': 1,
    'Jijel': 1,
    'Tébessa': 2,
    'Tipaza': 2,
    'Skikda': 1,
    'Khenchela': 1,
    'Guelma': 1,
    'Tizi Ouzou': 1
}

# 合并为 DataFrame
cbc_df = pd.DataFrame.from_dict(cbc_data, orient='index', columns=['CBC'])
clia_df = pd.DataFrame.from_dict(clia_data, orient='index', columns=['CLIA'])
merged_df = pd.concat([cbc_df, clia_df], axis=1).fillna(0).astype(int)
merged_df['Total'] = merged_df['CBC'] + merged_df['CLIA']
merged_df = merged_df.reset_index().rename(columns={'index': 'Province'})

# 绘图函数

def plot_total_map(shapefile_path: str, df_total: pd.DataFrame):
    gdf = gpd.read_file(shapefile_path)
    gdf['NAME_1'] = gdf['NAME_1'].astype(str).str.strip().str.title()

    # 特殊统一拼写
    df_total['Province'] = df_total['Province'].str.strip().str.replace("Sidi Bel Abbes", "Sidi Bel Abbès").str.title()

    merged = gdf.merge(df_total, left_on='NAME_1', right_on='Province', how='left')
    merged[['Total', 'CBC', 'CLIA']] = merged[['Total', 'CBC', 'CLIA']].fillna(0)
    merged = merged.to_crs(epsg=3395)
    merged['centroid'] = merged.geometry.centroid

    fig, ax = plt.subplots(figsize=(12, 10))
    merged.plot(
        column='Total',
        cmap='YlGnBu',
        linewidth=0.8,
        edgecolor='0.8',
        legend=True,
        ax=ax,
        missing_kwds={
            "color": "lightgrey",
            "edgecolor": "white",
            "hatch": "///",
            "label": "No data"
        }
    )

    # for idx, row in merged.iterrows():
    #     if row['Total'] >= 5:
    #         x, y = row['centroid'].x, row['centroid'].y
    #
    #         # 横向偏移
    #         if x < merged['centroid'].x.median():
    #             x_text = x - 200000
    #             align = 'right'
    #         else:
    #             x_text = x + 200000
    #             align = 'left'
    #
    #         # 垂直方向做交错排列（交替加减）
    #         offset = ((idx % 5) - 2) * 30000  # -60k, -30k, 0, +30k, +60k
    #         y_text = y + offset
    #
    #         label = f"{row['Province']}\n血球 {int(row['CBC'])} 发光 {int(row['CLIA'])}"
    #         ax.plot([x, x_text], [y, y_text], color='black', linewidth=0.5)
    #         ax.text(x_text, y_text, label, fontsize=7, ha=align, va='center')

    ax.set_title("阿尔及利亚业务板块", fontsize=16)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig("algeria_cbc_clia_labeled.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    shapefile_path = "gadm41_DZA_shp/gadm41_DZA_1.shp"
    plot_total_map(shapefile_path, merged_df)
