# Nomodit - Natural Language Text Modifier

Nomodit is a command-line tool (cli+tui) for linguistic refinement using a fine-tuned version of Google's Gemma 3n model, bringing language processing capabilities to run locally on most consumer hardware (>8GB RAM/VRAM) while fostering a learning environment through transparent reasoning on what changed and other forms of feedback. The model specializes in various language-related tasks including:

- Grammatical Error Correction (GEC)
- Text clarity improvement
- Coherence fixing
- Text neutralization
- Paraphrasing
- Text simplification

## Usage

### Command Line Interface
```
nomodit -i "instruction" "text to modify"
```

Example:
```
nomodit -i "Fix grammatical errors" "I has went to the store yesterday."
```

### Interactive TUI
Running `nomodit` without arguments launches an interactive text user interface with separate input areas for instructions and text.

![image](https://github.com/user-attachments/assets/af28d15b-41ee-4d64-85ce-20fa078e2a40)

### TUI Features

The interactive TUI provides features to help users understand and learn from text modifications.

- **Reasoning Traces**: View the model's step-by-step reasoning process for each modification, helping you understand why specific changes were made
- **Word-Level Diffs**: See precise word-by-word differences between original and modified text, making it easy to identify exactly what was changed
- These features work together to help users identify patterns in their writing mistakes and learn from the corrections.


## the LLM
nomodit is also the name of a series of fine-tuned version of Google's Gemma 3n model using the [CoEdit-cot-reasoning](https://huggingface.co/datasets/muzzz/coedit-cot-reasoning) dataset which augments the [CoEdit](https://huggingface.co/datasets/grammarly/coedit) dataset with reasoning traces to allow for nomodit models to have reasoning capabilities. 

The training runs for fine-tuning the models with unsloth's FastModel framework utlize PEFT techniques like QLoRA and add reasoning capabilites by using GRPO and rubrics as rewards for language tasks which are inherently non-verifiable.

You can find the subsequent notebooks for training the model, creating the reasoning traces using a larger llm and the benchmarking against other models on GEC [here](https://github.com/muzzlol/nomodit-training).
> **Note:** This work is currently in progress.