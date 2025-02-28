from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import ORPOConfig, ORPOTrainer
from datasets import load_dataset

from utils import get_device

def train(checkpoint="HuggingFaceTB/SmolLM2-135M", dataset_name:str="trl-lib/ultrafeedback_binarized", device=get_device()):

    print("Loading dataset...")

    ds = load_dataset(dataset_name)
    train_dataset = ds["train"]
    test_dataset = ds["test"]

    print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prpo_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    prpo_model.config.use_cache = False

    print("Starting training...")

    orpo_config = ORPOConfig()

    orpo_trainer = ORPOTrainer(
        model=prpo_model,
        args=orpo_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    orpo_trainer.train()

if __name__ == '__main__':
    pass