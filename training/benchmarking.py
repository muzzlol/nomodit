from llama_cpp import Llama
import os
import urllib.request
import subprocess
import tempfile
import re # For parsing errant output and tokenization functions
import time # For basic timing

# --- Configuration ---
MAX_EVAL_SAMPLES = 50 # Set to None to evaluate on the full dataset
LOG_SAMPLE_OUTPUTS = 5 # Number of original/corrected pairs to print per model

# --- Tokenization/Detokenization Functions (Added) ---
def detokenize_bea(text):
  """
  Converts BEA-style tokenized text (spaces before punctuation/contractions)
  back into natural text. (Using simpler, more reliable version)
  """
  # Normalize smart quotes/apostrophes before processing
  text = text.replace('“', '"').replace('”', '"').replace('’', "'")
  # Remove space before punctuation (commas, periods, question marks, exclamation marks, quotes)
  text = re.sub(r'\s+([.,?!"])', r'\1', text)
  # Join word + n't : " do n't " -> "don't"
  text = re.sub(r"(\w)\s+n't\b", r"\1n't", text)
  # Remove space before other common contractions ('s, 't, 'm, 'd, 'll, 've, 're)
  text = re.sub(r"\s+('(?:s|t|m|d|ll|ve|re))\b", r'\1', text)
  # Remove spaces around hyphens (e.g., " co - operations " -> "co-operations")
  text = re.sub(r'\s+-\s+', r'-', text)
  # Clean up potential double spaces created by replacements and strip
  text = re.sub(r'\s{2,}', ' ', text).strip()
  return text

def tokenize_like_bea(text):
  """
  Converts natural text into BEA-style tokenized text, adding spaces
  before punctuation and specific contractions/apostrophes.
  """
  # Normalize smart quotes/apostrophes before processing
  text = text.replace('“', '"').replace('”', '"').replace('’', "'")
  # --- Add initial spacing ---
  # Add spaces around hyphens first
  text = re.sub(r'-', r' - ', text)
  # --- Specific Contraction Handling (PRIORITY) ---
  # Handle "n't" (case‐insensitive) -> " n't"
  text = re.sub(r"n't\b", r" n't", text, flags=re.IGNORECASE)
  # Handle other common contractions -> " 's", " 'm", " 'll", etc.
  text = re.sub(r"([A-Za-z])('(?:s|m|d|ll|ve|re))\b", r"\1 \2", text)
  # --- General Punctuation/Quote Handling ---
  # Add space BEFORE punctuation [.,?!"] if preceded by non-space
  text = re.sub(r"(\S)([.,?!\"”])", r"\1 \2", text) # Excludes apostrophe now
  # Add space AFTER opening quote if followed by non-space
  text = re.sub(r'(["“])(\S)', r'\1 \2', text) # Added opening quote ”
  # --- Clean-up ---
  # Collapse multiple spaces into one
  text = re.sub(r'\s{2,}', ' ', text)
  # Strip leading/trailing spaces
  text = text.strip()
  return text

# --- Loading data ---

# Raw URL of the file
bea_dev = "https://raw.githubusercontent.com/grammarly/pillars-of-gec/main/data/evaluation_sets/bea-dev.txt"
bea_m2 = "https://raw.githubusercontent.com/grammarly/pillars-of-gec/refs/heads/main/data/evaluation_sets/bea-dev.m2"

folder = "data"
os.makedirs(folder, exist_ok=True)

dev_file_path = os.path.join(folder, "bea-dev.txt")
if not os.path.exists(dev_file_path):
    print(f"Downloading {bea_dev}...")
    urllib.request.urlretrieve(bea_dev, dev_file_path)
    print(f"File downloaded to: {dev_file_path}")
else:
    print(f"Using existing file: {dev_file_path}")


m2_file_path = os.path.join(folder, "bea-dev.m2")
if not os.path.exists(m2_file_path):
    print(f"Downloading {bea_m2}...")
    urllib.request.urlretrieve(bea_m2, m2_file_path)
    print(f"File downloaded to: {m2_file_path}")
else:
     print(f"Using existing file: {m2_file_path}")


# --- Utility Functions ---
def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f]

def create_subset_m2(input_m2_path, output_m2_path, num_sentences):
    """Reads an M2 file and writes the first num_sentences entries to a new file."""
    print(f"Creating subset M2 file ({num_sentences} sentences) at: {output_m2_path}")
    sentence_count = 0
    with open(input_m2_path, 'r', encoding='utf-8') as infile, \
         open(output_m2_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # Ensure line is not empty before checking prefix
            if line.strip() and line.startswith('S '):
                sentence_count += 1
                if sentence_count > num_sentences:
                    break
            # Check sentence_count before writing any line related to the current sentence block
            if sentence_count <= num_sentences and sentence_count > 0: # Only write if we are within the limit and started processing sentences
                 outfile.write(line)
            # Handle edge case where the loop might end before writing the last newline of the final sentence block
        if sentence_count <= num_sentences and sentence_count > 0:
             outfile.write('\n') # Ensure last entry ends properly if needed

    print(f"Finished creating subset M2 file.")


# --- Read Original Sentences (BEA Format) ---
all_original_sentences_bea = read_sentences(dev_file_path)
print(f"Read {len(all_original_sentences_bea)} total sentences from {dev_file_path}")

# Select subset for evaluation
if MAX_EVAL_SAMPLES is not None and MAX_EVAL_SAMPLES < len(all_original_sentences_bea):
    original_sentences_bea = all_original_sentences_bea[:MAX_EVAL_SAMPLES]
    print(f"Using a subset of {len(original_sentences_bea)} sentences for evaluation.")
else:
    original_sentences_bea = all_original_sentences_bea
    print(f"Using the full dataset of {len(original_sentences_bea)} sentences for evaluation.")


class LLM:
    def __init__(self, repo_id: str, filename: str = "*q8_0.gguf", verbose: bool = False, **kwargs):
        self.model_name = repo_id.split('/')[-1] # Extract model name for reporting
        print(f"Initializing {self.model_name}...")
        self.model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            verbose=verbose,
            **kwargs
        )
        print(f"{self.model_name} initialized.")

    # Takes NATURAL language prompts, returns NATURAL language results
    def __call__(self, prompts: list[str], system_prompt: str = "Correct any grammatical errors in the provided sentence. Only output the corrected sentence.", temperature: float = 0.1, max_tokens=150, **kwargs):
        results = []
        total_prompts = len(prompts)
        for i, prompt in enumerate(prompts):
            # Add progress indicator
            print(f"  Processing prompt {i+1}/{total_prompts}...", end='\r')
            try:
                response = self.model.create_chat_completion(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                # Handle potential empty responses or formatting issues
                if response['choices'] and 'message' in response['choices'][0] and 'content' in response['choices'][0]['message']:
                    correction = response['choices'][0]['message']['content'].strip()
                    # Basic check for empty string, could be more robust
                    if correction:
                         results.append(correction)
                    else:
                        print(f"\nWarning: Model generated empty output for prompt: {prompt[:80]}...") # Log snippet
                        results.append(prompt) # Use original sentence as fallback
                else:
                    print(f"\nWarning: Unexpected response structure or no choices for prompt: {prompt[:80]}...")
                    results.append(prompt) # Fallback to original
            except Exception as e:
                print(f"\nError during model inference for prompt: {prompt[:80]}...\n{e}")
                # Fallback to original sentence on error to avoid crashing evaluation
                results.append(prompt)
        print(f"\n  Finished processing {total_prompts} prompts.") # Newline after progress
        return results

# --- Model Initialization ---
# Using Q4_K_M for potentially faster eval runs

common_llm_params = {
    "n_gpu_layers": -1, # Use GPU acceleration if available
    "n_ctx": 512,       # Context window, might need adjustment for longer sentences
    "verbose": False    # Keep console cleaner
}

# Initialize models only once
# Ensure you have enough VRAM/RAM for the models you uncomment
llm_instances = {
    "gemma-3-4b-it": LLM("unsloth/gemma-3-4b-it-GGUF", filename="*Q4_K_M.gguf", **common_llm_params),
    "nomodit-4b": LLM("muzzz/nomodit-4b-merged-v0-Q4_K_M-GGUF", filename="*q4_k_m.gguf", verbose=False), # Assuming common params are ok
    # "phi-4-mini-instruct": LLM("unsloth/Phi-4-mini-instruct-GGUF", filename="*Q4_K_M.gguf", **common_llm_params),
    # "phi-4": LLM("unsloth/phi-4-GGUF", filename="*Q4_K_M.gguf", **common_llm_params),
    # "llama-3.1-8B-Instruct": LLM("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF", filename="*Q4_K_M.gguf", **common_llm_params),
    # "llama-3.2-3B-Instruct": LLM("unsloth/Llama-3.2-3B-Instruct-GGUF", filename="*Q4_K_M.gguf", **common_llm_params),
}


# --- Benchmarking ---

models_to_benchmark_names = [
    "gemma-3-4b-it",
    "nomodit-4b",
    # Uncomment the following models when ready
    # "phi-4-mini-instruct",
    # "phi-4", 
    # "llama-3.1-8B-Instruct", 
    # "llama-3.2-3B-Instruct",
]

results = {}

# Regex to capture P, R, F0.5 from errant_compare output
# Updated regex to handle the table format output
score_pattern = re.compile(
    r"=========== Span-Based Correction ============\n"  # Match header
    r"TP\s+FP\s+FN\s+Prec\s+Rec\s+F0\.5\n"             # Match table header row
    r"\d+\s+\d+\s+\d+\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)" # Capture Prec, Rec, F0.5 from data row
)

print("\n--- Starting Benchmark ---")

# --- Prepare Input Data --- #
# Detokenize the input sentences ONCE before the loop
print(f"Detokenizing {len(original_sentences_bea)} sentences for LLM input...")
natural_input_sentences = [detokenize_bea(s) for s in original_sentences_bea]
print("Detokenization complete.")

# --- Create Reference M2 Subset (if needed) --- #
reference_m2_to_use = m2_file_path
temp_ref_m2_subset_path = None # Initialize path variable

if MAX_EVAL_SAMPLES is not None and MAX_EVAL_SAMPLES < len(all_original_sentences_bea):
    # Create a temporary file path that persists until explicitly deleted
    temp_ref_m2_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".m2", encoding='utf-8')
    temp_ref_m2_subset_path = temp_ref_m2_file.name
    temp_ref_m2_file.close() # Close the file handle immediately, but keep the file
    create_subset_m2(m2_file_path, temp_ref_m2_subset_path, len(original_sentences_bea))
    reference_m2_to_use = temp_ref_m2_subset_path
    print(f"Using subset reference M2: {reference_m2_to_use}")
else:
    print(f"Using full reference M2: {reference_m2_to_use}")


for model_key in models_to_benchmark_names:
    llm = llm_instances.get(model_key)
    if not llm:
        print(f"Warning: Model key '{model_key}' not found in initialized instances. Skipping.")
        continue

    model_name = llm.model_name # Use name from LLM object for consistency
    print(f"\n===== Evaluating model: {model_name} =====")

    # Paths for temporary files specific to this model iteration
    temp_orig_bea_path = None
    temp_cor_bea_path = None
    temp_hyp_m2_path = None

    try:
        # --- Create temporary file for ORIGINAL sentences (BEA format) --- #
        # This file is needed as -orig input for errant_parallel
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt", encoding='utf-8') as temp_orig_file:
            for sentence in original_sentences_bea: # Write the BEA format sentences
                temp_orig_file.write(sentence + '\n')
            temp_orig_bea_path = temp_orig_file.name
        print(f"Created temp orig (BEA format): {temp_orig_bea_path}")


        # --- Run Model (using detokenized input) --- #
        start_time = time.time()
        print(f"Generating corrections for {len(natural_input_sentences)} sentences...")
        # Pass the NATURAL language sentences to the LLM
        corrected_sentences_natural = llm(natural_input_sentences)
        generation_time = time.time() - start_time
        print(f"Generation took: {generation_time:.2f} seconds")

        # --- Retokenize Output --- #
        print("Retokenizing LLM output to BEA format...")
        corrected_sentences_bea = [tokenize_like_bea(s) for s in corrected_sentences_natural]
        print("Retokenization complete.")


        # --- Log Sample Outputs --- #
        if LOG_SAMPLE_OUTPUTS > 0:
            print(f"\n--- Sample Outputs for {model_name} (First {min(LOG_SAMPLE_OUTPUTS, len(original_sentences_bea))}) ---")
            for i in range(min(LOG_SAMPLE_OUTPUTS, len(original_sentences_bea))):
                print(f"Orig (BEA)  [{i+1}]: {original_sentences_bea[i]}")
                # print(f"Input (Nat) [{i+1}]: {natural_input_sentences[i]}") # Optional: print detokenized input
                # print(f"Output (Nat)[{i+1}]: {corrected_sentences_natural[i]}") # Optional: print natural output
                print(f"Corr (BEA)  [{i+1}]: {corrected_sentences_bea[i]}")
                print("---")
            print("\n")

        # --- Evaluation --- #
        if len(corrected_sentences_bea) != len(original_sentences_bea):
            print(f"Error: Number of corrected sentences ({len(corrected_sentences_bea)}) does not match original ({len(original_sentences_bea)}) for model {model_name}. Skipping evaluation.")
            results[model_name] = {"P": "Error", "R": "Error", "F0.5": "Length Mismatch", "Generation Time (s)": f"{generation_time:.2f}", "Eval Time (s)": "N/A"}
            continue # Skip to the next model

        eval_start_time = time.time()

        # 3. Write RETOKENIZED CORRECTED sentences (BEA format) to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".cor.txt", encoding='utf-8') as temp_cor_file:
            for sentence in corrected_sentences_bea: # Write the BEA format sentences
                temp_cor_file.write(sentence + '\n')
            temp_cor_bea_path = temp_cor_file.name
        print(f"Created temp corrected (BEA format): {temp_cor_bea_path}")


        # 4. Create hypothesis M2 file using errant_parallel
        #    Input: -orig (BEA format), -cor (BEA format)
        #    Output: hypothesis M2 (ERRANT will likely re-tokenize internally based on spaCy,
        #            but we feed it BEA format for consistency at this step)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".hyp.m2") as temp_hyp_m2_file:
            temp_hyp_m2_path = temp_hyp_m2_file.name

        print(f"Running errant_parallel... Orig: {temp_orig_bea_path}, Cor: {temp_cor_bea_path}, Out: {temp_hyp_m2_path}")
        parallel_cmd = ["errant_parallel", "-orig", temp_orig_bea_path, "-cor", temp_cor_bea_path, "-out", temp_hyp_m2_path]
        parallel_result = subprocess.run(parallel_cmd, capture_output=True, text=True, check=False)

        if parallel_result.returncode != 0:
            print(f"Error running errant_parallel for {model_name}:")
            print(f"Stderr: {parallel_result.stderr}")
            results[model_name] = {"P": "Error", "R": "Error", "F0.5": "errant_parallel failed", "Generation Time (s)": f"{generation_time:.2f}", "Eval Time (s)": "N/A"}
            continue # Skip to next model
        if "Processing 0 sentences" in parallel_result.stderr: # Check if ERRANT processed anything
             print(f"Warning: errant_parallel reported processing 0 sentences. Check input files or ERRANT setup.")
             print(f"Stderr: {parallel_result.stderr}")
             # Decide whether to continue or mark as error

        # 5. Compare hypothesis M2 with reference M2 using errant_compare
        #    Input: -hyp (Generated M2), -ref (BEA Reference M2 - potentially subsetted)
        print(f"Running errant_compare... Hyp: {temp_hyp_m2_path}, Ref: {reference_m2_to_use}")
        compare_cmd = ["errant_compare", "-hyp", temp_hyp_m2_path, "-ref", reference_m2_to_use]
        compare_result = subprocess.run(compare_cmd, capture_output=True, text=True, check=False)

        if compare_result.returncode != 0:
            print(f"Error running errant_compare for {model_name}:")
            print(f"Stderr: {compare_result.stderr}")
            results[model_name] = {"P": "Error", "R": "Error", "F0.5": "errant_compare failed", "Generation Time (s)": f"{generation_time:.2f}", "Eval Time (s)": "N/A"}
            continue # Skip to next model

        # 6. Parse scores
        eval_time = time.time() - eval_start_time
        print(f"Evaluation took: {eval_time:.2f} seconds")
        # print(f"errant_compare output for {model_name}:\n{compare_result.stdout}") # Uncomment for debugging

        match = score_pattern.search(compare_result.stdout)
        if match:
            p, r, f05 = match.groups()
            print(f"Scores for {model_name} (on {len(original_sentences_bea)} samples): P={p}, R={r}, F0.5={f05}")
            results[model_name] = {
                "P": float(p),
                "R": float(r),
                "F0.5": float(f05),
                "Samples": len(original_sentences_bea),
                "Generation Time (s)": f"{generation_time:.2f}",
                "Eval Time (s)": f"{eval_time:.2f}"
            }
        else:
            print(f"Could not parse scores from errant_compare output for {model_name}.")
            print(f"Output:\n{compare_result.stdout}")
            results[model_name] = {"P": "Error", "R": "Error", "F0.5": "Parsing failed", "Samples": len(original_sentences_bea), "Generation Time (s)": f"{generation_time:.2f}", "Eval Time (s)": f"{eval_time:.2f}"}

    except Exception as e:
        print(f"An unexpected error occurred during evaluation for {model_name}: {e}")
        # Attempt to calculate eval time even if error occurred mid-evaluation
        current_eval_time = time.time() - (eval_start_time if 'eval_start_time' in locals() and eval_start_time else start_time)
        results[model_name] = {"P": "Error", "R": "Error", "F0.5": f"Exception: {type(e).__name__}", "Samples": len(original_sentences_bea), "Generation Time (s)": f"{generation_time:.2f}", "Eval Time (s)": f"{current_eval_time:.2f}"}
    finally:
        # silently remove temp files without logging
        for path in [temp_orig_bea_path, temp_cor_bea_path, temp_hyp_m2_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

# --- Final Cleanup for Subset Reference M2 ---
if temp_ref_m2_subset_path and os.path.exists(temp_ref_m2_subset_path):
    try:
        os.remove(temp_ref_m2_subset_path)
    except OSError:
        pass


print("\n--- Benchmark Results ---")
# Sort results alphabetically by model name for consistent output
sorted_results = dict(sorted(results.items()))

for model_name, scores in sorted_results.items():
    print(f"Model: {model_name}")
    # Define a preferred order for metrics if desired
    metric_order = ["F0.5", "P", "R", "Samples", "Generation Time (s)", "Eval Time (s)"]
    for metric in metric_order:
        if metric in scores:
            value = scores[metric]
            # Format floats nicely
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    # Print any other metrics not in the preferred order (e.g., error messages)
    for metric, value in scores.items():
        if metric not in metric_order:
             print(f"  {metric}: {value}")
    print("-" * 20)