import argparse
import transformers
import numpy as np


def tokenize_file(path, save_path):
    model_name = "THUDM/chatglm-6b"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True)
    with open(path, "r") as f_in:
        data = f_in.read()
        ids = tokenizer.encode(data)
        arr = np.array(ids)
        arr.tofile(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/k8s/test.txt")
    parser.add_argument("--save_path", type=str, default="data/k8s/test.bin")
    args = parser.parse_args()
    tokenize_file(args.data_path, args.save_path)


if __name__ == "__main__":
    main()
