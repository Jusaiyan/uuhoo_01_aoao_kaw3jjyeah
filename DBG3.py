import pandas as pd
import yfinance as yf
import sqlite3
from itertools import islice

# --- 設定値 ---
target_start_date = "2025-07-01"
target_end_date = "2025-08-01"
chunk_size = 200

# --- データ取得 ---
# CSVからティッカーを作成
try:
    df_codes = pd.read_csv("codes.csv", encoding="utf-8", dtype=str)
    tickers = [code.strip() + ".T" for code in df_codes["code"]]
except FileNotFoundError:
    print("Error: codes.csv not found.")
    exit()

# チャンク分割関数
def chunked(iterable, size):
    it = iter(iterable)
    return iter(lambda: list(islice(it, size)), [])

# データ取得
data_chunks = []
for chunk in chunked(tickers, chunk_size):
    df = yf.download(
        tickers=" ".join(chunk),
        start=target_start_date,
        end=target_end_date,
        auto_adjust=False,
        progress=False
    )
    data_chunks.append(df)

data = pd.concat(data_chunks, axis=1)

# --- DataFrame整形 ---
# マルチインデックスの整形
data.columns = data.columns.swaplevel(0, 1)
data = data.stack(level=0, future_stack=True).reset_index()
data.rename(columns={'level_1': 'Ticker'}, inplace=True)

# ここから追加 ----------------------------------
# Date列をYYYY-MM-DD形式に変換
data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
# -----------------------------------------------

# 列名をSQL安全な形に変換（スペースや記号をアンダースコアに）
data.columns = [c.replace(" ", "_").replace("-", "_") for c in data.columns]

# --- SQLiteへの保存 ---
# SQLiteデータベースに接続（存在しない場合は新規作成）
con = sqlite3.connect("stock_data.db")
cursor = con.cursor()

# DataFrameをテーブルに追記
data.to_sql("stock_prices", con, if_exists="append", index=False)

# 重複レコードを削除
cursor.execute("""
    DELETE FROM stock_prices
    WHERE rowid NOT IN (
        SELECT MIN(rowid)
        FROM stock_prices
        GROUP BY Date, Ticker
    );
""")

# 変更をコミット
con.commit()

# 接続を閉じる
con.close()

print("データの保存、追記、重複削除が完了しました。")
