import os
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer

ds = load_dataset("michaelwzhu/ChatMed_Consult_Dataset", split="train").shuffle(seed=42).train_test_split(test_size=0.5)["train"]
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat")

def process_func(example):
    MAX_LENGTH = 256
    input_ids, attention_mask, labels = [], [], []
    query = tokenizer("\nHuman: " + example["query"].strip() + "\n\nAssistant: ")
    response = tokenizer(example["response"] + tokenizer.eos_token)
    input_ids = query["input_ids"] + response["input_ids"]
    attention_mask = query["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(query["input_ids"]) + response["input_ids"]
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

model = AutoModelForCausalLM.from_pretrained("/data/jasonzhou/Models/Qwen1.5-0.5B-Chat")
config = LoraConfig(task_type=TaskType.CAUSAL_LM, target_modules=["20.self_attn.q_proj", "20.self_attn.k_proj", "20.self_attn.v_proj"], modules_to_save=["word_embeddings"])
model = get_peft_model(model, config)

model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="./rui20240425",
    per_device_train_batch_size=16,
    logging_steps=10,
    num_train_epochs=1,
    learning_rate=5e-5,
    overwrite_output_dir=True,
    local_rank=int(os.environ.get('LOCAL_RANK', -1)),
    dataloader_num_workers=4,
    save_strategy = "steps", 
    save_steps = 1000,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()

# model = model.cuda()
# ipt = tokenizer("Human: {}\n{}".format("考试有哪些技巧？", "").strip() + "\n\nAssistant: ", return_tensors="pt").to(model.device)
# tokenizer.decode(model.generate(**ipt, max_length=128, do_sample=True)[0], skip_special_tokens=True)

# python -m torch.distributed.launch --nproc_per_node=4 train.py