import pandas as pd
import matplotlib.pyplot as plt

def analyze_bias(file_path):
    # Read the .inter file (tab-separated)
    df = pd.read_csv(file_path, sep='\t', header=0)
    
    # Item frequency
    item_counts = df['item_id:token'].value_counts()
    
    # Statistical summary
    print(f"Total interactions: {len(df)}")
    print(f"Total unique items: {len(item_counts)}")
    print(f"Top 10 items (Popularity Bias):\n{item_counts.head(10)}")
    print(f"Bottom 10 items:\n{item_counts.tail(10)}")
    
    # Percentage of interactions for top 10% items
    top_10_percent_count = int(len(item_counts) * 0.1)
    top_10_percent_interactions = item_counts.head(top_10_percent_count).sum()
    print(f"\nTop 10% items account for {top_10_percent_interactions / len(df) * 100:.2f}% of interactions.")

if __name__ == "__main__":
    analyze_bias(r'd:\UIT\HK6\MXH\DoAn\Recbole-Debias\dataset\ml\ml.train.inter')
