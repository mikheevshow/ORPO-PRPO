from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import ORPOConfig
from datasets import load_dataset

from scripts.prpo import PRPOTrainer
from  utils import get_device

def train(checkpoint:str="HuggingFaceTB/SmolLM2-135M",
          learning_rate: float=8e-6,
          per_device_train_batch_size:int=4,
          per_device_eval_batch_size: int=4,
          gradient_accumulation_steps: int=4,
          beta: float=0.1,
          num_train_epochs:int=4,
          dataset_name:str="trl-lib/ultrafeedback_binarized",
          device:str=get_device()):

    print("Loading dataset...")

    ds = load_dataset(dataset_name)
    train_dataset = ds["train"]
    test_dataset = ds["test"]

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.pad_token_id = tokenizer.eos_token_id

    prpo_model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
    prpo_model.config.use_cache = False

    prpo_training_args = ORPOConfig(output_dir=f"./models/SMOL-PRPO-bs{per_device_train_batch_size}-ga-{gradient_accumulation_steps}",
                                    learning_rate=learning_rate,
                                    lr_scheduler_type="linear",
                                    beta=beta,
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

    prpo_train = PRPOTrainer(
        model=prpo_model,
        processing_class=tokenizer,
        args=prpo_training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    prpo_train.train()

    print("Finish training.")