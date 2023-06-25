import json
import argparse
import logging
from tqdm import tqdm
import os
from datasets import load_dataset

logger = logging.getLogger(__name__)


def convert_text_to_jsonl(data_path, save_path, max_seq_length):
    with open(data_path, 'r') as f_in, open(save_path, 'w', encoding='utf-8') as f_out:
        content = f_in.read()
        chunk_size = max_seq_length  # 设置分块大小
        chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
        for chunk in tqdm(chunks, desc="formatting.."):
            data = {
                "context":"",
                "target": str(chunk),
            }
            json.dump(data, f_out, ensure_ascii=False)
            f_out.write('\n')


def main():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/k8s/test.txt")
    parser.add_argument("--save_path", type=str, default="data/k8s/test.jsonl")
    parser.add_argument("--max_seq_length", type=int, default=2000)

    args = parser.parse_args()
    logging.info("args:{}".format(args))

    convert_text_to_jsonl(args.data_path, args.save_path, args.max_seq_length)


if __name__ == "__main__":
    main()
