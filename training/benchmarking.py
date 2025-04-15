from llama_cpp import Llama
import spacy
import errant
import os
import urllib.request
import subprocess
import tempfile
import re # For parsing errant output
import time # For basic timing

# --- Configuration ---
MAX_EVAL_SAMPLES = 50 # Set to None to evaluate on the full dataset
LOG_SAMPLE_OUTPUTS = 5 # Number of original/corrected pairs to print per model

# loading data

# Raw URL of the file
bea_dev = "https://raw.githubusercontent.com/grammarly/pillars-of-gec/main/data/evaluation_sets/bea-dev.txt"
bea_m2 = "https://raw.githubusercontent.com/grammarly/pillars-of-gec/refs/heads/main/data/evaluation_sets/bea-dev.m2"

folder = "data"
os.makedirs(folder, exist_ok=True)

dev_file_path = os.path.join(folder, "bea-dev.txt")
urllib.request.urlretrieve(bea_dev, dev_file_path)

m2_file_path = os.path.join(folder, "bea-dev.m2")
urllib.request.urlretrieve(bea_m2, m2_file_path)

print(f"File downloaded to: {dev_file_path}")
print(f"File downloaded to: {m2_file_path}")


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
            if line.startswith('S '):
                sentence_count += 1
                if sentence_count > num_sentences:
                    break
            if sentence_count <= num_sentences:
                outfile.write(line)
    print(f"Finished creating subset M2 file.")


# --- Read Original Sentences ---
all_original_sentences = read_sentences(dev_file_path)
print(f"Read {len(all_original_sentences)} total sentences from {dev_file_path}")

# Select subset for evaluation
if MAX_EVAL_SAMPLES is not None and MAX_EVAL_SAMPLES < len(all_original_sentences):
    original_sentences = all_original_sentences[:MAX_EVAL_SAMPLES]
    print(f"Using a subset of {len(original_sentences)} sentences for evaluation.")
else:
    original_sentences = all_original_sentences
    print(f"Using the full dataset of {len(original_sentences)} sentences for evaluation.")


class LLM:
    def __init__(self, repo_id: str, filename: str = "*q8_0.gguf", verbose: bool = False, **kwargs):
        self.model_name = repo_id.split('/')[-1] # Extract model name for reporting
        self.model = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            verbose=verbose,
            **kwargs
        )

    def __call__(self, prompts: list[str], system_prompt: str = "Correct any grammatical errors in the provided sentence. Only output the corrected sentence.", temperature: float = 0.1, max_tokens=150, **kwargs):
        results = []
        for prompt in prompts:
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
                        print(f"Warning: Model generated empty output for prompt: {prompt}")
                        # Use original sentence as output if model fails? Or a placeholder?
                        # Using original for now to allow ERRANT processing, though score will be low.
                        results.append(prompt)
                else:
                    print(f"Warning: Unexpected response structure or no choices for prompt: {prompt}")
                    results.append(prompt) # Fallback to original
            except Exception as e:
                print(f"Error during model inference for prompt: {prompt}\n{e}")
                # Fallback to original sentence on error to avoid crashing evaluation
                results.append(prompt)
        return results

# --- Model Initialization ---
# Using Q4_K_M for potentially faster eval runs

gemma3_4b = LLM(
    "unsloth/gemma-3-4b-it-GGUF",
    filename="*Q4_K_M.gguf",
    n_gpu_layers=-1, # Use GPU acceleration if available
    n_ctx=512,       # Context window
    verbose=False    # Keep console cleaner
)

# we will test with these aswell once we get everything working
phi4_mini = LLM(
    "unsloth/Phi-4-mini-instruct-GGUF",
    filename="*Q4_K_M.gguf",
    n_gpu_layers=-1, # Use GPU acceleration if available
    n_ctx=512,       # Context window
    verbose=False    # Keep console cleaner
)

phi4 = LLM(
    "unsloth/phi-4-GGUF",
    filename="*Q4_K_M.gguf",
    n_gpu_layers=-1, # Use GPU acceleration if available
    n_ctx=512,       # Context window
    verbose=False    # Keep console cleaner
)

llama3_8b = LLM(
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
    filename="*Q4_K_M.gguf",
    n_gpu_layers=-1, # Use GPU acceleration if available
    n_ctx=512,       # Context window
    verbose=False    # Keep console cleaner
)

llama3_3b = LLM(
    "unsloth/Llama-3.2-3B-Instruct-GGUF",
    filename="*Q4_K_M.gguf",
    n_gpu_layers=-1, # Use GPU acceleration if available
    n_ctx=512,       # Context window
    verbose=False    # Keep console cleaner
)

# --- Benchmarking ---

models_to_benchmark = [
    gemma3_4b,
    phi4_mini,
    # phi4, # Uncomment when ready
    # llama3_8b, # Uncomment when ready
    # llama3_3b, # Uncomment when ready
]

results = {}

# Regex to capture P, R, F0.5 from errant_compare output
score_pattern = re.compile(r"Precision:\s*(\d+\.\d+)\nRecall:\s*(\d+\.\d+)\nF0\.5:\s*(\d+\.\d+)")

print("\n--- Starting Benchmark ---")

for llm in models_to_benchmark:
    model_name = llm.model_name
    print(f"\nEvaluating model: {model_name}")

    # Prepare temporary file paths for subset data
    temp_orig_subset_path = None
    temp_ref_m2_subset_path = None
    temp_cor_path = None
    temp_hyp_m2_path = None

    try:
        # --- Create temporary files for the SUBSET of data --- #
        # 1. Subset of Original Sentences
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt", encoding='utf-8') as temp_orig_file:
            for sentence in original_sentences:
                temp_orig_file.write(sentence + '\n')
            temp_orig_subset_path = temp_orig_file.name

        # 2. Subset of Reference M2 (Only if evaluating a subset)
        if MAX_EVAL_SAMPLES is not None and MAX_EVAL_SAMPLES < len(all_original_sentences):
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".m2", encoding='utf-8') as temp_m2_ref_file:
                temp_ref_m2_subset_path = temp_m2_ref_file.name
            create_subset_m2(m2_file_path, temp_ref_m2_subset_path, len(original_sentences))
            reference_m2_to_use = temp_ref_m2_subset_path
        else:
            reference_m2_to_use = m2_file_path # Use the original full M2 if not subsetting

        # --- Run Model --- #
        start_time = time.time()
        print(f"Generating corrections for {len(original_sentences)} sentences...")
        corrected_sentences = llm(original_sentences)
        generation_time = time.time() - start_time
        print(f"Generation took: {generation_time:.2f} seconds")

        # --- Log Sample Outputs --- #
        if LOG_SAMPLE_OUTPUTS > 0:
            print(f"\n--- Sample Outputs for {model_name} (First {min(LOG_SAMPLE_OUTPUTS, len(original_sentences))}) ---")
            for i in range(min(LOG_SAMPLE_OUTPUTS, len(original_sentences))):
                print(f"Orig [{i+1}]: {original_sentences[i]}")
                print(f"Corr [{i+1}]: {corrected_sentences[i]}")
                print("---")
            print("\n")

        # --- Evaluation --- #
        if len(corrected_sentences) != len(original_sentences):
            print(f"Error: Number of corrected sentences ({len(corrected_sentences)}) does not match original ({len(original_sentences)}) for model {model_name}. Skipping evaluation.")
            results[model_name] = {"P": "Error", "R": "Error", "F0.5": "Length Mismatch", "Generation Time (s)": f"{generation_time:.2f}", "Eval Time (s)": "N/A"}
            continue # Skip to the next model

        eval_start_time = time.time()

        # 3. Write CORRECTED sentences (subset) to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt", encoding='utf-8') as temp_cor_file:
            for sentence in corrected_sentences:
                temp_cor_file.write(sentence + '\n')
            temp_cor_path = temp_cor_file.name

        # 4. Create hypothesis M2 file using errant_parallel (using subset orig and subset cor)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".m2") as temp_hyp_m2_file:
            temp_hyp_m2_path = temp_hyp_m2_file.name

        print(f"Running errant_parallel... Orig: {temp_orig_subset_path}, Cor: {temp_cor_path}, Out: {temp_hyp_m2_path}")
        parallel_cmd = ["errant_parallel", "-orig", temp_orig_subset_path, "-cor", temp_cor_path, "-out", temp_hyp_m2_path]
        parallel_result = subprocess.run(parallel_cmd, capture_output=True, text=True, check=False)

        if parallel_result.returncode != 0:
            print(f"Error running errant_parallel for {model_name}:")
            print(f"Stderr: {parallel_result.stderr}")
            results[model_name] = {"P": "Error", "R": "Error", "F0.5": "errant_parallel failed", "Generation Time (s)": f"{generation_time:.2f}", "Eval Time (s)": "N/A"}
            continue # Skip to next model

        # 5. Compare hypothesis M2 with reference M2 using errant_compare (using subset ref M2)
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
            print(f"Scores for {model_name} (on {len(original_sentences)} samples): P={p}, R={r}, F0.5={f05}")
            results[model_name] = {
                "P": float(p),
                "R": float(r),
                "F0.5": float(f05),
                "Samples": len(original_sentences),
                "Generation Time (s)": f"{generation_time:.2f}",
                "Eval Time (s)": f"{eval_time:.2f}"
            }
        else:
            print(f"Could not parse scores from errant_compare output for {model_name}.")
            print(f"Output:\n{compare_result.stdout}")
            results[model_name] = {"P": "Error", "R": "Error", "F0.5": "Parsing failed", "Samples": len(original_sentences), "Generation Time (s)": f"{generation_time:.2f}", "Eval Time (s)": f"{eval_time:.2f}"}

    except Exception as e:
        print(f"An unexpected error occurred during evaluation for {model_name}: {e}")
        eval_time = time.time() - (eval_start_time if 'eval_start_time' in locals() else start_time) # Estimate eval time
        results[model_name] = {"P": "Error", "R": "Error", "F0.5": f"Exception: {e}", "Samples": len(original_sentences), "Generation Time (s)": f"{generation_time:.2f}", "Eval Time (s)": f"{eval_time:.2f}"}
    finally:
        # 7. Clean up temporary files
        print("Cleaning up temporary files...")
        for path in [temp_orig_subset_path, temp_ref_m2_subset_path, temp_cor_path, temp_hyp_m2_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"  Removed: {path}")
                except OSError as e:
                    print(f"  Error removing {path}: {e}")


print("\n--- Benchmark Results ---")
for model_name, scores in results.items():
    print(f"Model: {model_name}")
    for metric, value in scores.items():
        print(f"  {metric}: {value}")
    print("-" * 20)
