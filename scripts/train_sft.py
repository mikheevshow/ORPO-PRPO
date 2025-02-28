from transformers import AutoTokenizer, AutoModelWithLMHead
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


def train(checkpoint:str="HuggingFaceTB/SmolLM2-135M",
          dataset_name:str="trl-lib/ultrafeedback_binarized",
          learning_rate:float=5e-5,
          max_steps:int=1_000,
          per_device_train_batch_size:int=4,):

    ds = load_dataset(dataset_name)
    train_dataset, test_dataset = ds["train"], ds["test"]

    train_dataset = train_dataset.map(lambda x: {'messages': x['chosen']},
                                 remove_columns=['chosen', 'rejected', 'score_chosen', 'score_rejected'])
    test_dataset = test_dataset.map(lambda x: {'messages': x['chosen']},
                               remove_columns=['chosen', 'rejected', 'score_chosen', 'score_rejected'])

    print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelWithLMHead.from_pretrained(checkpoint)

    print("Training...")

    sft_config = SFTConfig(
        learning_rate=learning_rate,
        output_dir="./models/SMOL-SFT",
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=4,
        logging_steps=1,
        evaluation_strategy="steps",
        eval_steps=50,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()

    print("Finish training.")