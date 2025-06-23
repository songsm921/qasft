from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import datasets
from transformers import TrainingArguments
import os
from accelerate import Accelerator
def train_phase(model, tokenizer, pruned_solutions, iteration, args, acc):
    model.train()
    accelerator = acc
    model = model.to(accelerator.device)
    if args.system_prompt:
        instruction_template = "<\uff5cUser\uff5c>"
        response_template = "<think>\n"
    else:
        instruction_template = None
        response_template = "Solution\n"
    collator = DataCollatorForCompletionOnlyLM(
        instruction_template = instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    train_dataset = datasets.Dataset.from_list(pruned_solutions)
    if args.quant:
        training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, f"quant_model_iteration_{iteration}"),
        num_train_epochs=args.train_epoch,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        logging_steps=10,
        learning_rate=1e-5,
        weight_decay=1e-2,
        warmup_ratio=0.1,
        # lr_scheduler_type = 'cosine',
        save_only_model=True,
        bf16=True,
        remove_unused_columns=True,
        )
        # training_args = TrainingArguments(
        # output_dir=os.path.join(args.output_dir, f"quant_model_iteration_{iteration}"),
        # num_train_epochs=args.train_epoch,
        # per_device_train_batch_size=1,
        # gradient_accumulation_steps=1,
        # logging_steps=10,
        # learning_rate=1e-5,
        # weight_decay=1e-4,
        # warmup_ratio=0.05,
        # lr_scheduler_type = 'cosine',
        # save_only_model=True,
        # bf16=True,
        # remove_unused_columns=True,
        # )
    else:
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"tpt_model_iteration_{iteration}"),
            num_train_epochs=args.train_epoch,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            logging_steps=10,
            learning_rate=1e-5,
            weight_decay=0.01,
            warmup_ratio=0.1,
            bf16=True,
            remove_unused_columns=True,
        )
    training_args.dataset_text_field = 'text'
    training_args.max_seq_length = 32768
    trainer = SFTTrainer(
        model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collator
    )
    # print('LINE 59')
    trainer.train()
    # print('train')
    if args.quant:
        save_path = os.path.join(args.output_dir, f"quant_model_iteration_{iteration}")
    else:
        save_path = os.path.join(args.output_dir, f"tpt_model_iteration_{iteration}")
    trainer.save_model(output_dir=save_path)
    tokenizer.save_pretrained(save_path)
    trainer.accelerator.wait_for_everyone()
    return