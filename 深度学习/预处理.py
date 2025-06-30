# Author: moqiHe
# Date: 2025-06-25
# Description: 
from docx import Document
import os

# def docx_to_txt(docx_path, txt_path):
#     doc = Document(docx_path)
#     with open(txt_path, "w", encoding="utf-8") as f:
#         for para in doc.paragraphs:
#             if para.text.strip():
#                 f.write(para.text.strip() + "\n")
#
# # 设置文件夹路径
# docx_folder = "./新能源汽车维修故障案例"  # 你的docx文件夹路径
# txt_folder = "./txt"    # 输出的txt文件夹路径
# os.makedirs(txt_folder, exist_ok=True)
#
# for filename in os.listdir(docx_folder):
#     if filename.endswith(".docx"):
#         docx_path = os.path.join(docx_folder, filename)
#         txt_path = os.path.join(txt_folder, filename.replace(".docx", ".txt"))
#         docx_to_txt(docx_path, txt_path)
#
# with open("cleaned.txt", "w", encoding="utf-8") as fw:
#     for f in os.listdir(txt_folder):
#         if f.endswith(".txt") and not f.startswith("~$"):
#             with open(os.path.join(txt_folder, f), encoding="utf-8") as fr:
#                 lines = fr.readlines()
#                 for line in lines:
#                     line = line.strip()
#                     if line:
#                         fw.write(line + "\n")

import os

input_folder = "./txt"
output_file = "cleaned_for_doccano.txt"

max_len = 1024 # 每条最多字符数
buffer = []

def clean_line(line):
    return line.strip().replace("\n", "").replace("\r", "")

with open(output_file, "w", encoding="utf-8") as fw:
    for fname in os.listdir(input_folder):
        if fname.endswith(".txt") and not fname.startswith("~$"):
            with open(os.path.join(input_folder, fname), "r", encoding="utf-8") as fr:
                for line in fr:
                    line = clean_line(line)
                    if not line:
                        continue
                    buffer.append(line)
                    if sum(len(s) for s in buffer) > max_len:
                        fw.write(" ".join(buffer) + "\n")
                        buffer = []
    # 写入最后一段
    if buffer:
        fw.write(" ".join(buffer) + "\n")