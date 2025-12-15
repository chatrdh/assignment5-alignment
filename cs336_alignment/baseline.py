import json
import torch
from datasets import Dataset
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn  
import pandas as pd


# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DATASET_NAME = "hiyouga/math12k"
OUTPUT_FILE = "evaluation_results.jsonl"
TEST_SIZE = 2048  # Number of examples to evaluate

# R1-Zero Prompt Template 
R1_PROMPT_TEMPLATE = """A conversation between User and Assistant. The User asks a question, and the Assistant solves it. 
The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer.
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags,
respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.

User: {question}
Assistant: <think>  """

def load_data():
    subsets = [
        'algebra',
        'counting_and_probability',
        'geometry',
        'intermediate_algebra',
        'number_theory',
        'prealgebra',
        'precalculus'
    ]

    # Base URL for Parquet files
    base_url = "https://huggingface.co/datasets/EleutherAI/hendrycks_math/resolve/refs%2Fconvert%2Fparquet"

    # Load and combine all test examples
    test_dfs = []
    for subset in subsets:
        print(f"Loading {subset} test split...")
        test_url = f"{base_url}/{subset}/test/0000.parquet"
        df = pd.read_parquet(test_url)
        df['subset'] = subset  # Add a column to track which subset each row came from
        test_dfs.append(df)

    test_data = pd.concat(test_dfs, ignore_index=True)
    ds_test = Dataset.from_pandas(test_data)
    return ds_test

def evaluate_baseline():
    # 1. Prepare Data
    dataset = load_data()
    
    # 2. Prepare Prompts
    prompts = [R1_PROMPT_TEMPLATE.format(question=ex['problem']) for ex in dataset]
    ground_truths = [ex['answer'] for ex in dataset]

    # 3. Initialize Model (Optimized for T4 GPU)
    print("Initializing Model...")
    llm = LLM(
        model=MODEL_NAME, 
        dtype="half",                 # Required for T4 (float16)
        gpu_memory_utilization=0.90,  # Maximize VRAM usage
        enforce_eager=True           
    )

    # 4. Define Sampling Params [cite: 141-144]
    # Stop exactly at </answer> to prevent rambling
    sampling_params = SamplingParams(
        temperature=1,              # Slight creativity allowed for math reasoning
        max_tokens=1024,              # Allow long chains of thought
        stop=["</answer>"], 
        include_stop_str_in_output=True
    )

    # 5. Generate
    print("Generating responses...")
    outputs = llm.generate(prompts, sampling_params)

    # 6. Grading Loop
    print("Grading responses...")
    results = []
    correct_count = 0
    format_count = 0

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        gt = ground_truths[i]
        
        # Use Dr. GRPO reward function
        # This checks both formatting (<answer> tags) and math correctness
        scores = r1_zero_reward_fn(generated_text, gt)
        
        is_correct = scores['answer_reward'] == 1.0
        is_formatted = scores['format_reward'] == 1.0
        
        if is_correct: correct_count += 1
        if is_formatted: format_count += 1
        
        # Log result
        results.append({
            "problem": dataset[i]['problem'],
            "ground_truth": gt,
            "generated_text": generated_text,
            "format_reward": scores['format_reward'],
            "answer_reward": scores['answer_reward']
        })

    # 7. Metrics & Saving
    accuracy = correct_count / len(outputs)
    format_compliance = format_count / len(outputs)
    
    print("-" * 40)
    print(f"Total Evaluated: {len(outputs)}")
    print(f"Format Compliance: {format_compliance:.2%}")
    print(f"Math Accuracy:     {accuracy:.2%}")
    print("-" * 40)
    
    # Save to disk 
    with open(OUTPUT_FILE, "w") as f:
        for res in results:
            f.write(json.dumps(res) + "\n")
    print(f"Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    evaluate_baseline()