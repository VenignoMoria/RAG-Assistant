import argparse
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", default="artifacts/router-qlora")
    parser.add_argument("--output", default="artifacts/router-merged")
    args = parser.parse_args()
    # Merge в полноразмерные веса; нельзя делать merge в training NF4 weights.
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32, device_map="cpu")
    model = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    model.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)


if __name__ == "__main__":
    main()
