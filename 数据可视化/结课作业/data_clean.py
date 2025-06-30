
import pandas as pd
import os

# 所有年份文件路径
years = ['2015', '2016', '2017', '2018', '2019']
file_dir = './archive'
all_data = []

for year in years:
    path = os.path.join(file_dir, f"{year}.csv")
    df = pd.read_csv(path)

    if year in ['2015', '2016']:
        df_clean = pd.DataFrame({
            "Country or region": df["Country"],
            "Score": df["Happiness Score"],
            "GDP per capita": df["Economy (GDP per Capita)"],
            "Social support": df["Family"],
            "Healthy life expectancy": df["Health (Life Expectancy)"],
            "Freedom to make life choices": df["Freedom"],
            "Generosity": df["Generosity"],
            "Perceptions of corruption": df["Trust (Government Corruption)"]
        })
    elif year == '2017':
        df_clean = pd.DataFrame({
            "Country or region": df["Country"],
            "Score": df["Happiness.Score"],
            "GDP per capita": df["Economy..GDP.per.Capita."],
            "Social support": df["Family"],
            "Healthy life expectancy": df["Health..Life.Expectancy."],
            "Freedom to make life choices": df["Freedom"],
            "Generosity": df["Generosity"],
            "Perceptions of corruption": df["Trust..Government.Corruption."]
        })
    elif year in ['2018', '2019']:
        df_clean = pd.DataFrame({
            "Country or region": df["Country or region"],
            "Score": df["Score"],
            "GDP per capita": df["GDP per capita"],
            "Social support": df["Social support"],
            "Healthy life expectancy": df["Healthy life expectancy"],
            "Freedom to make life choices": df["Freedom to make life choices"],
            "Generosity": df["Generosity"],
            "Perceptions of corruption": df["Perceptions of corruption"]
        })

    df_clean["Year"] = int(year)
    all_data.append(df_clean)

# 合并所有年份数据
df_all = pd.concat(all_data)

# 计算每个国家的7列指标平均值
df_avg = df_all.groupby("Country or region").mean(numeric_only=True).reset_index()

# 保存输出
df_avg.to_csv("happiness_2015_2019_avg.csv", index=False)
print("已保存为 happiness_2015_2019_avg.csv")