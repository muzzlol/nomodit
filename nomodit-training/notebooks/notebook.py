# !pip install unsloth vllm

# !pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

from unsloth import FastModel

import torch

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    max_seq_length = 256, # Reduced based on data analysis (99th percentile 132)
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False,
    full_finetuning = False,
)

print(model.is_loaded_in_4bit)
print(model.is_quantized)
print(model.quantization_method)
'''
True
True
True
'''

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 16,           # Larger = higher accuracy, but no significant improvements relative to compute inc. check LoRA paper
    lora_alpha = 32,  # Recommended alpha == r at least
    lora_dropout = 0.1,
    bias = "none",
    random_state = 3407,
)

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)


from datasets import load_dataset
try:
    train_ds = load_dataset("grammarly/coedit", split="train")
    val_ds = load_dataset("grammarly/coedit", split="validation")
except ValueError as e:
    print(f"Caught ValueError, likely need to specify config. Trying 'default'. Error: {e}")
    train_ds = load_dataset("grammarly/coedit", name="default", split="train")
    val_ds = load_dataset("grammarly/coedit", name="default", split="validation")

print(train_ds)
print(val_ds)

"""o/p:
Dataset({
    features: ['_id', 'task', 'src', 'tgt'],
    num_rows: 69071
})
Dataset({
    features: ['_id', 'task', 'src', 'tgt'],
    num_rows: 1712
})
"""

def format_dataset_with_template(example, tokenizer):

    src_txt = example["src"]
    tgt_txt = example["tgt"]

    # 1. Prepare the conversation history in the required format
    messages = [
        {"role": "user", "content": src_txt},
        {"role": "model", "content": tgt_txt},
    ]

    # 2. Apply the chat template
    try:
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    except Exception as e:
        print(f"Error applying chat template to example: {example}")
        print(f"Error: {e}")
        formatted_text = "" # Return empty string on error to avoid crashing map

    # Apply the logic from remove_special_tokens found in Unsloth code
    if formatted_text.startswith(tokenizer.bos_token):
        formatted_text = formatted_text[len(tokenizer.bos_token):]

    # 3. Return in the desired dictionary format for .map
    return {"text": formatted_text}

processed_train_ds = train_ds.map(
    lambda example: format_dataset_with_template(example, tokenizer),
    batched=False,
    num_proc=1 
)
processed_val_ds = val_ds.map(
    lambda example: format_dataset_with_template(example, tokenizer),
    batched=False,
    num_proc=1
)

print(processed_train_ds)
print(processed_val_ds)

"""o/p:
Dataset({
    features: ['_id', 'task', 'src', 'tgt', 'text'],
    num_rows: 69071
})
Dataset({
    features: ['_id', 'task', 'src', 'tgt', 'text'],
    num_rows: 1712
})"""

# removing original columns
columns_to_remove = list(train_ds.features)
print(f"\nRemoving original columns: {columns_to_remove}")
processed_train_ds = processed_train_ds.remove_columns(columns_to_remove)
processed_val_ds = processed_val_ds.remove_columns(columns_to_remove)

print("\nProcessed train dataset sample (using tokenizer template):")
print(processed_train_ds)
if len(processed_train_ds) > 0:
    print(processed_train_ds[0]["text"])
else:
     print("Processed training dataset is empty or first example failed.")


print("\nProcessed validation dataset sample (using tokenizer template):")
print(processed_val_ds)
if len(processed_val_ds) > 0:
    print(processed_val_ds[0]['text'])
else:
    print("Processed validation dataset is empty or first example failed.")


"""o/p:

Processed train dataset sample (using tokenizer template):
Dataset({
    features: ['text'],
    num_rows: 69071
})
<start_of_turn>user
Remove all grammatical errors from this text: For example, countries with a lot of deserts can terraform their desert to increase their habitable land and using irrigation to provide clean water to the desert.<end_of_turn>
<start_of_turn>model
For example, countries with a lot of deserts can transform their desert to increase their habitable land and use irrigation to provide clean water to the desert.<end_of_turn>


Processed validation dataset sample (using tokenizer template):
Dataset({
    features: ['text'],
    num_rows: 1712
})
<start_of_turn>user
Paraphrase this sentence: Why are you arresting me?<end_of_turn>
<start_of_turn>model
Why am I being arrested?<end_of_turn>
"""

'''
### Train the model
Now let's use Huggingface TRL's `SFTTrainer`! More docs here: [TRL SFT docs](https://huggingface.co/docs/trl/sft_trainer).
'''
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=processed_train_ds,
    eval_dataset=processed_val_ds,
    args=SFTConfig(
        output_dir="./checkpoints256",
        dataset_text_field="text",
        max_seq_length=256,
        packing=True,
        per_device_train_batch_size=64,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        num_train_epochs=1,
        learning_rate=3e-5,
        logging_steps=25,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
    )
)


print(trainer.train_dataset)

"""We also use Unsloth's `train_on_completions` method to only train on the assistant outputs and ignore the loss on the user's inputs. This helps increase accuracy of finetunes!"""

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

print(trainer.train_dataset)

print(trainer.train_dataset[100]["text"])
print()
print(trainer.train_dataset[100]["input_ids"])
print()
print(trainer.train_dataset[100]["attention_mask"])
print()
print(trainer.train_dataset[100]["labels"])


"""o/p:
<start_of_turn>user
Fix grammar in this sentence: If engineers do not come up with new ideas, they cannot find best solution for the problems.<end_of_turn>
<start_of_turn>model
If engineers do not come up with new ideas, they cannot find the best solution for different problems.<end_of_turn>


[105, 2364, 107, 36819, 40095, 528, 672, 13315, 236787, 1637, 22072, 776, 711, 2229, 872, 607, 861, 6549, 236764, 901, 3914, 1586, 1791, 3465, 573, 506, 4078, 236761, 106, 107, 105, 4368, 107, 2859, 22072, 776, 711, 2229, 872, 607, 861, 6549, 236764, 901, 3914, 1586, 506, 1791, 3465, 573, 1607, 4078, 236761, 106, 107]

[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 2859, 22072, 776, 711, 2229, 872, 607, 861, 6549, 236764, 901, 3914, 1586, 506, 1791, 3465, 573, 1607, 4078, 236761, 106, 107]
"""

total_training_tokens = 0
for i in range(trainer.train_dataset.num_rows):
    total_training_tokens += len(trainer.train_dataset[i]["input_ids"])
    
print("Total training tokens in CoEdIT dataset:", total_training_tokens)

"""o/p:
Total training tokens in CoEdIT dataset: 3994943
"""

import numpy as np

lengths = [len(x) for x in trainer.train_dataset['input_ids']]
print(f"Token lengths stats:")
print(f"Min: {np.min(lengths)}")
print(f"Max: {np.max(lengths)}")
print(f"Mean: {np.mean(lengths)}")
print(f"Median: {np.median(lengths)}")
print(f"95th percentile: {np.percentile(lengths, 95)}")
print(f"99th percentile: {np.percentile(lengths, 99)}")

""""o/p:
Token lengths stats:
Min: 17
Max: 686
Mean: 57.83820995786944
Median: 51.0
95th percentile: 107.0
99th percentile: 132.0
"""

# @title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

"""Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`"""

trainer_stats = trainer.train()

# @title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


messages = [{
    "role": "user",
    "content": [{"type" : "text", "text" : "Simplify the sentence: You'll earn a lot as an engineer.",}]
}]
# potential answer : As an engineer, you learn a lot.
text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt = True, # Must add for generation
)

from transformers import TextStreamer
_ = model.generate(
    **tokenizer([text], return_tensors = "pt").to("cuda"),
    max_new_tokens = 512, # Increase for longer outputs!
    # Recommended Gemma-3 settings!
    temperature = 0.1,
    streamer = TextStreamer(tokenizer, skip_prompt = True),
)

if True:
    hf_repo_name = "muzzz/nomodit-4b-merged" 
    # token w/ write access
    hf_token = ""
    
    print(f"Attempting to push merged model to: {hf_repo_name}")
    try:
        model.push_to_hub_merged(
            hf_repo_name, tokenizer,
            token = hf_token,
            commit_message = "Upload fine-tuned Gemma-3 4B (merged float16)"
        )
        print(f"Successfully pushed merged model to {hf_repo_name}")
    except Exception as e:
        print(f"Error pushing merged model to Hugging Face Hub: {e}")
        print("Please ensure you have the correct repo name, write token, and network connectivity.")
