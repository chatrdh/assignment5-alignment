

import pandas as pd
from datasets import Dataset

# Use the same URL as above
data_url = "https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4/resolve/main/data/train-00000-of-00001.parquet"
# 1. Load into Pandas
df = pd.read_parquet(data_url)

# 2. Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

print("Success! Loaded via Pandas.")
print(dataset[0])



