import pandas as pd

def calculate_top_100_bias(file_path):
    df = pd.read_csv(file_path, sep='\t', header=0)
    item_counts = df['item_id:token'].value_counts()
    top_100_interactions = item_counts.head(100).sum()
    total_interactions = len(df)
    percentage = (top_100_interactions / total_interactions) * 100
    print(f"Top 100 items account for {top_100_interactions} interactions.")
    print(f"Total interactions: {total_interactions}")
    print(f"Percentage: {percentage:.2f}%")

if __name__ == "__main__":
    calculate_top_100_bias(r'd:\UIT\HK6\MXH\DoAn\Recbole-Debias\dataset\ml\ml.train.inter')
