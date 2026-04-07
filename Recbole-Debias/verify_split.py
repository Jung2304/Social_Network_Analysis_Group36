import pandas as pd
import os

def check_intervene_mask(file_path):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    df = pd.read_csv(file_path, sep='\t', header=0)
    counts = df['intervene_mask:token'].value_counts()
    print(f"\nStats for {os.path.basename(file_path)}:")
    print(counts)
    print(f"Total: {len(df)}")

if __name__ == "__main__":
    base_path = r'd:\UIT\HK6\MXH\DoAn\Recbole-Debias\dataset\ml'
    check_intervene_mask(os.path.join(base_path, 'ml.train.inter'))
    check_intervene_mask(os.path.join(base_path, 'ml.valid.inter'))
    check_intervene_mask(os.path.join(base_path, 'ml.test.inter'))
