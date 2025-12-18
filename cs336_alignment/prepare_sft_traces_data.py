import json
from datasets import Dataset
import pandas as pd

def create_sft_dataset():
    print("Downloading OpenR1-Math-220k dataset...")
    # This dataset contains DeepSeek R1 reasoning traces for Math problems
    data_url = "https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/resolve/main/data/train-00000-of-00010.parquet"
    df = pd.read_parquet(data_url)

    ds = Dataset.from_pandas(df)

    output_file = "sft.jsonl"
    print(f"Processing and saving to {output_file}...")

    with open(output_file, "w", encoding="utf-8") as f:
        # We'll take the first 5,000 examples to save time (or remove [:5000] for all)
        for i, row in enumerate(ds):
            if i >= 5000: break 
            
            # 1. Get the Question
            question = row["problem"]

            # 2. Get the Reasoning Trace + Answer
            # The dataset often has multiple generations; we pick the first one.
            # DeepSeek R1 traces usually already include <think> tags.
            response = row["generations"][0]

            # 3. Create the prompt format expected by the model
            # The assignment usually asks for the standard "User/Assistant" format
            prompt = (
                f"A conversation between User and Assistant. The user asks a question, "
                f"and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\n"
                f"User: {question}\n"
                f"Assistant: "
            )

            # 4. Construct the JSON object
            entry = {
                "prompt": prompt,
                "response": response # Contains <think>... and answer
            }
            
            f.write(json.dumps(entry) + "\n")

    print("Done! You can now point your SFT script to 'sft.jsonl'")

if __name__ == "__main__":
    create_sft_dataset()