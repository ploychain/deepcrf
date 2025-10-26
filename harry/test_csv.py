import pandas as pd

# 读取 CSV
csv_path = '/home/harry/deepcfr/equity_table.csv'
try:
    df = pd.read_csv(csv_path, dtype={'hand': str, 'equity': float}, skipinitialspace=True)
    print(f"CSV columns: {df.columns}")
    print(f"Total rows: {len(df)}")
    print(f"First 5 rows:\n{df.head()}")
    # 测试特定手牌
    test_hands = ['42o', 'AAs', 'AKs']
    for hand in test_hands:
        try:
            equity = df.loc[df['hand'] == hand, 'equity'].values[0]
            print(f"Equity for {hand}: {equity}")
        except IndexError:
            print(f"Equity for {hand}: Not found")
except Exception as e:
    print(f"Error reading CSV: {e}")