import argparse
import text
from utils import load_filepaths_and_text

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="清洗文本并生成新的 filelist")
    parser.add_argument(
        "--out_extension",
        default="cleaned",
        help="输出文件的扩展名，默认 .cleaned"
    )
    parser.add_argument(
        "--text_index",
        default=1,
        type=int,
        help="filelist 中文本所在的列索引，默认 1"
    )
    parser.add_argument(
        "--filelists",
        nargs="+",
        required=True,
        help="需要清洗的 filelist 文件列表"
    )
    parser.add_argument(
        "--text_cleaners",
        nargs="+",
        default=["chinese_cleaners"],
        help="文本清洗器列表，默认中文清洗器"
    )

    args = parser.parse_args()

    # 遍历所有 filelist 文件
    for filelist in args.filelists:
        print("START:", filelist)
        # 加载文件路径和文本，返回列表 [(audio_path, text), ...]
        filepaths_and_text = load_filepaths_and_text(filelist)

        # 遍历每条数据，清洗文本
        for i in range(len(filepaths_and_text)):
            original_text = filepaths_and_text[i][args.text_index]
            cleaned_text = text._clean_text(original_text, args.text_cleaners)
            filepaths_and_text[i][args.text_index] = cleaned_text

        # 写入新的 filelist 文件
        new_filelist = filelist + "." + args.out_extension
        with open(new_filelist, "w", encoding="utf-8") as f:
            f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

        print("DONE:", new_filelist)

if __name__ == "__main__":
    main()
