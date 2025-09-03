import argparse
import random
import text
from utils import load_filepaths_and_text

def main():
    parser = argparse.ArgumentParser(description="清洗文本并生成新的 filelist，同时拆分训练/验证集")
    parser.add_argument("--out_extension", default="cleaned", help="输出文件扩展名")
    parser.add_argument("--text_index", default=1, type=int, help="文本列索引")
    parser.add_argument("--filelists", nargs="+", required=True, help="需要清洗的 filelist 文件列表")
    parser.add_argument("--text_cleaners", nargs="+", default=["chinese_cleaners"], help="文本清洗器列表")
    parser.add_argument("--val_ratio", default=0.1, type=float, help="验证集占比，默认 0.1")
    args = parser.parse_args()

    for filelist in args.filelists:
        print("START:", filelist)
        filepaths_and_text = load_filepaths_and_text(filelist)

        # 清洗文本
        for i in range(len(filepaths_and_text)):
            original_text = filepaths_and_text[i][args.text_index]
            cleaned_text = text._clean_text(original_text, args.text_cleaners)
            filepaths_and_text[i][args.text_index] = cleaned_text

        # 打乱顺序
        random.shuffle(filepaths_and_text)

        # 拆分训练集和验证集
        val_size = int(len(filepaths_and_text) * args.val_ratio)
        val_lines = filepaths_and_text[:val_size]
        train_lines = filepaths_and_text[val_size:]

        # 写入训练集文件
        train_file = filelist + "." + args.out_extension + "_train"
        with open(train_file, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in train_lines])

        # 写入验证集文件
        val_file = filelist + "." + args.out_extension + "_val"
        with open(val_file, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in val_lines])

        print(f"DONE! 训练集: {len(train_lines)} 条, 验证集: {len(val_lines)} 条")
        print("训练集文件:", train_file)
        print("验证集文件:", val_file)

if __name__ == "__main__":
    main()
