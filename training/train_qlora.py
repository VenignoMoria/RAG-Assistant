import argparse
from pathlib import Path
import mlflow
import torch
import yaml
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from trl import SFTConfig, SFTTrainer
from .prepare_dataset import prepare


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA example requires a compatible CUDA GPU and bitsandbytes")
    set_seed(config["seed"])
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=config["quantization"]["load_in_4bit"],
                                     bnb_4bit_quant_type=config["quantization"]["quant_type"],
                                     bnb_4bit_use_double_quant=config["quantization"]["double_quant"],
                                     bnb_4bit_compute_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(config["model"], quantization_config=quantization,
                                               device_map={"": torch.cuda.current_device()}, torch_dtype=dtype)
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    hyper = config["training"]
    lora = LoraConfig(r=config["lora"]["r"], lora_alpha=config["lora"]["alpha"],
                      lora_dropout=config["lora"]["dropout"], target_modules=config["lora"]["target_modules"],
                      task_type="CAUSAL_LM", bias="none")
    train = prepare(config["train_file"], tokenizer, hyper["max_length"])
    val = prepare(config["val_file"], tokenizer, hyper["max_length"])
    settings = SFTConfig(output_dir=config["output_dir"], learning_rate=hyper["learning_rate"],
                         num_train_epochs=hyper["epochs"], per_device_train_batch_size=hyper["microbatch"],
                         per_device_eval_batch_size=1, gradient_accumulation_steps=hyper["gradient_accumulation"],
                         max_length=hyper["max_length"], warmup_ratio=hyper["warmup_ratio"],
                         lr_scheduler_type=hyper["scheduler"], seed=config["seed"],
                         bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
                         completion_only_loss=True, packing=False, gradient_checkpointing=True,
                         eval_strategy="epoch", save_strategy="epoch", logging_steps=1, report_to="none")
    trainer = SFTTrainer(model=model, args=settings, train_dataset=train, eval_dataset=val,
                         processing_class=tokenizer, peft_config=lora)
    # Проверяем реальную маску первого batch: prompt=-100, completion обучается.
    batch = next(iter(trainer.get_train_dataloader()))
    labels = batch["labels"][0]
    if not (labels == -100).any() or not (labels != -100).any():
        raise RuntimeError("Completion loss mask is invalid")
    mlflow.set_experiment("support-copilot-router")
    with mlflow.start_run(run_name="qlora"):
        mlflow.log_params({"model": config["model"], "seed": config["seed"], "lora_r": config["lora"]["r"],
                           "learning_rate": hyper["learning_rate"], "epochs": hyper["epochs"]})
        mlflow.log_artifact(args.config)
        output = trainer.train()
        mlflow.log_metrics(output.metrics)
        mlflow.log_metrics(trainer.evaluate())
        trainer.save_model(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    main()
