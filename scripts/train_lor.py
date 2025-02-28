from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import ORPOConfig

from scripts.lor import LORTrainer

def train(checkpoint:[str | AutoModelForCausalLM],
          tokenizer: Optional[AutoTokenizer]=None,
          learning_rate: float=8e-6,
          per_device_train_batch_size: int=4,
          per_device_eval_batch_size: int=4,
          gradient_accumulation_steps: int=4,
          num_train_epochs: int=4,
          dataset_name:str="trl-lib/ultrafeedback_binarized"):
    if type(checkpoint) is str:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForCausalLM.from_pretrained(checkpoint)
    else:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided")
        model = checkpoint

    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    print("loading dataset")

    ds = load_dataset(dataset_name)
    train_dataset = ds['train']
    test_dataset = ds['test']

    lor_config = ORPOConfig(
        output_dir=f"./models/SMOL-SFT-LOR-bs{per_device_train_batch_size}-ga-{gradient_accumulation_steps}",
        learning_rate=learning_rate,
        lr_scheduler_type="linear",
        beta=1,  # <- set to 1 permanently
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        evaluation_strategy="steps",
        bf16=True,
        do_eval=True,
        eval_steps=100,
        logging_steps=100,
        warmup_steps=10,
        report_to="none")

    trainer = LORTrainer(
        model=model,
        processing_class=tokenizer,
        args=lor_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
