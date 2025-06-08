# nomodit

# Nomodit - Natural Language Text Modifier

Nomodit is a command-line tool for text editing and modification using a fine-tuned version of Google's Gemma model. The tool specializes in various language-related tasks including:

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


## Technical Details
Built on a fine-tuned version of Google's Gemma 3 model using the CoEdit dataset, which contains paired examples of original and improved text across multiple english language editing tasks. 
