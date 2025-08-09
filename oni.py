# app.py (Streamlit 用完全版)
def show_main_page3():
    import streamlit as st
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import datetime
    from itertools import islice
    from dateutil.relativedelta import relativedelta

    st.set_page_config(page_title="成長株 初動スクリーニング", layout="wide")

    # --- ユーザーUI: 日付・上位表示数など ---
    today = datetime.date.today()
    one_month_ago = today - relativedelta(months=1)

    st.title("成長株 初動サイン スクリーニング（現物取引用）")
    st.caption("注意：本ツールは参考用です。最終判断は自己責任でお願いします。")

    col1, col2 = st.columns([1, 1])
    with col1:
        A = st.date_input("開始日 (YYYY-MM-DD)", value=one_month_ago)
    with col2:
        B = st.date_input("終了日 (YYYY-MM-DD)", value=today)

    top_n = st.number_input("上位何件を表示するか (騰落率降順)", min_value=1, max_value=500, value=30, step=1)

    # パラメータ（必要ならユーザーに公開する）
    st.sidebar.header("スクリーニング パラメータ")
    ema_period = st.sidebar.number_input("EMA 期間", min_value=5, max_value=200, value=21)
    pivot_lookback = st.sidebar.number_input("ピボット（過去何日）", min_value=10, max_value=120, value=40)
    rebound_days = st.sidebar.number_input("EMA割れから何日以内に上抜けで反発とみなすか", min_value=1, max_value=10, value=5)
    test_days = st.sidebar.number_input("シグナル後の検証日数 (営業日)", min_value=5, max_value=30, value=14)
    volume_multiplier = st.sidebar.number_input("出来高増加閾値 (x倍)", min_value=1.0, max_value=5.0, value=1.5, step=0.1)

    # CSV からティッカーリスト（JPX一覧を codes.csv に置く想定）
    st.sidebar.header("データ設定")
    #codes_csv_path = st.sidebar.text_input("codes.csv のパス", value="codes.csv")
    codes_csv_path = "codes.csv"

    run_button = st.button("実行 ▶︎ スクリーニング開始")

    if not run_button:
        st.info("開始日・終了日とパラメータを設定後、[実行] を押してください。")
        return

    # --- データ取得（yfinance 一回だけ） ---
    st.info("データを yfinance から取得します（一度だけ）。ティッカー数が多いと時間かかります。")
    try:
        df_codes = pd.read_csv(codes_csv_path, encoding="utf-8", dtype=str)
        tickers = [code.strip() + ".T" for code in df_codes["code"]]
    except Exception as e:
        st.error(f"codes.csv の読み込みに失敗しました: {e}")
        return

    # chunk helper
    def chunked(iterable, size):
        it = iter(iterable)
        return iter(lambda: list(islice(it, size)), [])

    target_start_date = str(A)
    target_end_date = str(B)

    chunk_size = 200  # yfinance に渡す数（安全側）

    data_chunks = []
    failed_chunks = 0
    with st.spinner("yfinance からデータ取得中..."):
        for chunk in chunked(tickers, chunk_size):
            try:
                df = yf.download(
                    tickers=" ".join(chunk),
                    start=target_start_date,
                    end=target_end_date,
                    auto_adjust=False,  # Adjusted を別列で使う
                    progress=False
                )
                # skip empty
                if not df.empty:
                    data_chunks.append(df)
            except Exception as e:
                failed_chunks += 1
                # continue silently (ログ出力は下で)
                continue

    if not data_chunks:
        st.error("取得できるデータがありませんでした。日付範囲や codes.csv を確認してください。")
        return

    if failed_chunks > 0:
        st.warning(f"{failed_chunks} 個のチャンクでデータ取得に失敗しました（無視して続行）。")

    # マージして single multiindex データに
    data = pd.concat(data_chunks, axis=1)  # columns: (('Open','7203.T'),('High','7203.T'),...)
    # 重要: yfinance は 'Adj Close' 列を返す。計算は 'Adj Close' を主に使う（分割考慮）
    # ただしピボット高値は 'High' を用いる（調整済み high を使いたければ別途調整処理が必要）
    required_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    # check presence
    present_tickers = []
    for col in data.columns.levels[1]:
        # some tickers may have partial columns; check for 'Adj Close' presence
        try:
            _ = data[("Adj Close", col)]
            present_tickers.append(col)
        except Exception:
            continue

    if not present_tickers:
        st.error("データに 'Adj Close' 列が見つかりません。yfinance の取得結果を確認してください。")
        return

    st.success(f"{len(present_tickers)} 銘柄のデータを取得しました。解析を開始します。")

    # --- 解析: 銘柄ごとに指標算出 ---
    results = []
    progress_bar = st.progress(0)
    total = len(present_tickers)
    for idx, ticker in enumerate(present_tickers):
        progress_bar.progress(int((idx / total) * 100))
        try:
            # 個別DFを切り出し (列レベル2がティッカー)
            # Use Adj Close for price-based calculations (return, EMA)
            df_t = pd.DataFrame({
                "Open": data[("Open", ticker)],
                "High": data[("High", ticker)],
                "Low": data[("Low", ticker)],
                "Close": data[("Close", ticker)],
                "AdjClose": data[("Adj Close", ticker)],
                "Volume": data[("Volume", ticker)]
            }).dropna(how='all')

            if df_t.empty or len(df_t) < 10:
                continue

            # 指標計算: EMA21 (AdjClose ベース)
            df_t["EMA"] = df_t["AdjClose"].ewm(span=ema_period, adjust=False).mean()

            # pivot_high = 過去 pivot_lookback 日の High の最大（当日を含まないブレイク判定のため shift)
            df_t["pivot_high"] = df_t["High"].rolling(window=pivot_lookback, min_periods=1).max()

            # 20日平均出来高（当日を含まない比較が自然なので shift）
            df_t["vol20"] = df_t["Volume"].rolling(window=20, min_periods=1).mean()

            # EMA割れ -> rebound_days 日以内のどこかで AdjClose が EMA を上回るか判定
            df_t["below_ema"] = df_t["AdjClose"] < df_t["EMA"]
            df_t["touch_or_break"] = False

            # 過去行ループ（効率は簡易実装。大量銘柄ではベクトル化の最適化を検討）
            for i in range(len(df_t)):
                if df_t["below_ema"].iat[i]:
                    # i の次から rebound_days のウィンドウを見て、AdjClose > EMA があれば True
                    start = i + 1
                    end = min(i + 1 + rebound_days, len(df_t))
                    window = df_t.iloc[start:end]
                    if not window.empty and (window["AdjClose"].values > window["EMA"].values).any():
                        df_t.at[df_t.index[i], "touch_or_break"] = True

            # 反発開始（rebound_start）: ある日の AdjClose が EMA を上回り、当日終値が前日終値より上、かつ前日が touch_or_break
            df_t["rebound_start"] = (
                (df_t["AdjClose"] > df_t["EMA"]) &
                (df_t["AdjClose"] > df_t["AdjClose"].shift(1)) &
                (df_t["touch_or_break"].shift(1) == True)
            )

            # ピボットブレイク: 当日の AdjClose が（過去 pivot_lookback 日の High の max, shift(1)）を上回ること
            df_t["pivot_high_shift"] = df_t["pivot_high"].shift(1)
            df_t["pivot_break"] = df_t["AdjClose"] > df_t["pivot_high_shift"]

            # 出来高急増: 当日の出来高 > volume_multiplier * vol20
            df_t["volume_spike"] = df_t["Volume"] > (volume_multiplier * df_t["vol20"])

            # 最終的な買いシグナル: rebound_start & pivot_break & volume_spike
            df_t["buy_signal"] = df_t["rebound_start"] & df_t["pivot_break"] & df_t["volume_spike"]

            # シミュレーション: シグナル日に対して test_days 以内の最大騰落率（予測）
            df_t["max_gain_within_testdays_pred"] = np.nan
            for i in range(len(df_t)):
                if df_t["buy_signal"].iat[i]:
                    start_price = df_t["AdjClose"].iat[i]
                    start_idx = i + 1
                    end_idx = min(i + 1 + test_days, len(df_t))
                    future_window = df_t["AdjClose"].iloc[start_idx:end_idx]
                    if not future_window.empty:
                        max_price = future_window.max()
                        max_gain = (max_price - start_price) / start_price * 100.0
                        df_t.iat[i, df_t.columns.get_loc("max_gain_within_testdays_pred")] = max_gain

            # 集計: 最新日がシグナルの場合・あるいはシグナルが存在する日をすべて出力するか選べる（ここではシグナルがある最終日にフォーカス）
            # ここでは、"最新日"にシグナルが立っている銘柄、もしくは期間内にシグナルが一度でも立った銘柄を採用する選択にできるが、
            # 目的が「現物で買う」なら直近シグナルが重要 -> 最新日シグナルを優先表示する
            # 最新にシグナルがあるか判定
            if df_t["buy_signal"].iat[-1]:
                signal_idx = df_t.index[-1]
                signal_row = df_t.iloc[-1]
                # 期間の始値 / 終値を使った騰落率（解析用）
                period_start_price = df_t["AdjClose"].iat[0]
                period_end_price = df_t["AdjClose"].iat[-1]
                pct_change_period = (period_end_price - period_start_price) / period_start_price * 100.0

                results.append({
                    "ticker": ticker.replace(".T", ""),
                    "signal_date": signal_idx.strftime("%Y-%m-%d"),
                    "last_close": float(period_end_price),
                    "period_pct_change": float(pct_change_period),
                    "ema21": float(signal_row["EMA"]),
                    "pivot_high": float(signal_row["pivot_high_shift"]) if not np.isnan(signal_row["pivot_high_shift"]) else None,
                    "volume_today": int(signal_row["Volume"]),
                    "vol20": float(signal_row["vol20"]),
                    "volume_spike": bool(signal_row["volume_spike"]),
                    "max_gain_2w_pred_pct": float(signal_row["max_gain_within_testdays_pred"]) if not np.isnan(signal_row["max_gain_within_testdays_pred"]) else np.nan,
                    "buy_signal": True
                })
            else:
                # 期間内に1件でもシグナルがあった場合、最新のシグナル行を取り出してヒストリカル候補として表示（任意）
                if df_t["buy_signal"].any():
                    # 直近のシグナル日
                    last_sig_idx = df_t.index[df_t["buy_signal"]].max()
                    last_sig_row = df_t.loc[last_sig_idx]
                    period_start_price = df_t["AdjClose"].iat[0]
                    period_end_price = df_t["AdjClose"].iat[-1]
                    pct_change_period = (period_end_price - period_start_price) / period_start_price * 100.0

                    results.append({
                        "ticker": ticker.replace(".T", ""),
                        "signal_date": last_sig_idx.strftime("%Y-%m-%d") + " (過去シグナル)",
                        "last_close": float(period_end_price),
                        "period_pct_change": float(pct_change_period),
                        "ema21": float(last_sig_row["EMA"]),
                        "pivot_high": float(last_sig_row["pivot_high_shift"]) if not np.isnan(last_sig_row["pivot_high_shift"]) else None,
                        "volume_today": int(last_sig_row["Volume"]),
                        "vol20": float(last_sig_row["vol20"]),
                        "volume_spike": bool(last_sig_row["volume_spike"]),
                        "max_gain_2w_pred_pct": float(last_sig_row["max_gain_within_testdays_pred"]) if not np.isnan(last_sig_row["max_gain_within_testdays_pred"]) else np.nan,
                        "buy_signal": True
                    })
                # それ以外はスルー

        except Exception:
            # 個別銘柄で何か失敗しても続行
            continue

    progress_bar.progress(100)

    if not results:
        st.warning("期間内に買いシグナルを満たす銘柄は見つかりませんでした。")
        return

    # DataFrame に整形して表示
    out_df = pd.DataFrame(results)

    # 騰落率降順でソート（period_pct_change）
    out_df = out_df.sort_values("period_pct_change", ascending=False).reset_index(drop=True)

    # 上位 top_n
    out_df = out_df.head(int(top_n))

    # 表示用にフォーマット
    out_df["period_pct_change"] = out_df["period_pct_change"].map(lambda x: f"{x:.2f}%")
    out_df["last_close"] = out_df["last_close"].map(lambda x: f"{x:,.0f} 円" if x >= 1 else f"{x:.2f}")
    out_df["ema21"] = out_df["ema21"].map(lambda x: f"{x:,.0f} 円" if pd.notnull(x) else "-")
    out_df["pivot_high"] = out_df["pivot_high"].map(lambda x: f"{x:,.0f} 円" if pd.notnull(x) else "-")
    out_df["vol20"] = out_df["vol20"].map(lambda x: f"{x:,.0f}" if pd.notnull(x) else "-")
    out_df["volume_today"] = out_df["volume_today"].map(lambda x: f"{x:,.0f}")
    out_df["volume_spike"] = out_df["volume_spike"].map(lambda v: "Yes" if v else "No")
    out_df["max_gain_2w_pred_pct"] = out_df["max_gain_2w_pred_pct"].map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")

    # 想定購入額の簡易表示（1株・100株）
    out_df["想定購入額(1株)"] = out_df["last_close"].apply(lambda x: x.replace(" 円","").replace(",","")).astype(float)
    out_df["想定購入額(100株)"] = out_df["想定購入額(1株)"] * 100
    out_df["想定購入額(1株)"] = out_df["想定購入額(1株)"].map(lambda x: f"{x:,.0f} 円")
    out_df["想定購入額(100株)"] = out_df["想定購入額(100株)"].map(lambda x: f"{x:,.0f} 円")

    # 最終表示カラム順
    display_cols = [
        "ticker", "signal_date", "last_close", "想定購入額(1株)", "想定購入額(100株)",
        "period_pct_change", "ema21", "pivot_high", "volume_today", "vol20", "volume_spike",
        "max_gain_2w_pred_pct"
    ]
    display_df = out_df[display_cols].rename(columns={
        "ticker": "銘柄",
        "signal_date": "シグナル日",
        "last_close": "終了日終値",
        "period_pct_change": "対象期間騰落率",
        "ema21": "EMA21",
        "pivot_high": "ピボット高値(直前)",
        "volume_today": "当日出来高",
        "vol20": "出来高20日平均",
        "volume_spike": "出来高急増?",
        "max_gain_2w_pred_pct": "2週間最大騰落率（予測）"
    })

    st.subheader(f"検出結果（上位 {len(display_df)} 件を表示）")
    st.dataframe(display_df, use_container_width=True)

    # 説明セクション
    with st.expander("表示項目の説明（押すと展開）", expanded=False):
        st.markdown("""
        **銘柄**: JPXコード（.T を外したもの）  
        **シグナル日**: buy_signal が検出された日（最新シグナル。過去シグナルの可能性がある場合は注記される）  
        **終了日終値**: 指定期間の最終日の調整終値（現物購入価格の参考）  
        **想定購入額(1株／100株)**: それぞれ終値を基に計算した簡易購入金額（手数料・税は含まない）  
        **対象期間騰落率**: 指定期間（開始日→終了日）の騰落率（%）  
        **EMA21**: 計算された EMA (期間 = sidebar で指定) の値  
        **ピボット高値(直前)**: ピボットとして計算した過去 N 日の最高値（直前の値）  
        **当日出来高 / 出来高20日平均**: 出来高の目安（当日と直近平均）  
        **出来高急増?**: 当日出来高が (volume_multiplier × 20日平均) を超えているか（Yes/No）  
        **2週間最大騰落率（予測）**: シグナル日に続く test_days (sidebar で指定) 日以内に記録された最高値を起点に算出した最大上昇率（過去実績ベースのシミュレーション値。将来の保証ではありません）
        """)

    st.success("解析完了。上の表は現物購入の参考になります。購入前に板・板寄せ・流動性・出来高・指値戦略を必ず確認してください。")

# Streamlit 実行用エントリ (このファイルを直接 run する時)
#if __name__ == "__main__":
#    show_main_page3()