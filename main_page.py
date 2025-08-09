def show_main_page():
    import streamlit as st #streamlit
    import pandas as pd            #pip install pandas
    import yfinance as yf          #pip insatll yfinance
    import datetime  #pip insatll datetime たぶん標準ライブラリ
    from itertools import islice   #標準ライブラリ インストール必要なし
    from dateutil.relativedelta import relativedelta #pip install python-dateutil
    
    # 本日を取得
    today = datetime.date.today()
    # 本日から1ヶ月前の日付を計算
    one_month_ago = today - relativedelta(months=1)
    
    # CSV からティッカーリストを作成
    #https://www.jpx.co.jp/markets/statistics-equities/misc/01.html　から一覧のCSVを取得すること。
    df_codes = pd.read_csv("codes.csv", encoding="utf-8", dtype=str)
    tickers = [code.strip() + ".T" for code in df_codes["code"]]

    # フラグ初期化
    if 'submitted' not in st.session_state:
        st.session_state.submitted = 0

    st.write("YYYY-MM-DD形式で指定")
    col1, col2 = st.columns(2)
    with col1:
        A = st.date_input("開始日",value=one_month_ago)
    with col2:
        B = st.date_input("終了日", value=today)

    nani=st.text_input("上位何位まで表示するか整数で",key="nani",value=30)
    def submit_action():
        st.session_state.submitted = 1
    def submit_clear():
        st.session_state.submitted = 0

    col3, col4 = st.columns(2)
    with col3:
        st.button("OK", on_click=submit_action,icon="✔️")
    with col4:
        st.button("CLEAR", on_click=submit_clear,icon="🗑️")

    if st.session_state.submitted == 1:
        # --- ユーザー設定可能パラメータ ---
        target_start_date=f"{A}" #開始日
        target_end_date=f"{B}" #終了日
        top_n =int(f"{nani}")    #上位何位まで表示するか
        # １回の download() に渡すティッカー上限数 こいつはいじらない方が安全 yahooの仕様により変更になる可能性あり。
        chunk_size = 200
        # ----------------------------------------
        # ティッカーリストをチャンクに分割するジェネレータ
        def chunked(iterable, size):
            it = iter(iterable)
            return iter(lambda: list(islice(it, size)), [])

        # 各チャンクごとにダウンロード → リストに格納
        data_chunks = []
        for chunk in chunked(tickers, chunk_size):
            try:
                # chunk は最大 chunk_size 長のリスト
                # yfinance は内部でさらに分割することもありますが、ここで手動チャンクしておく
                df = yf.download(
                    tickers=" ".join(chunk),
                    start=target_start_date,
                    end=target_end_date,
                    auto_adjust=False,
                    progress=False
                )
                data_chunks.append(df)
            except Exception as e:
                pass

        # チャンクごとの DataFrame を列方向でマージ（マルチインデックス保持）
        data = pd.concat(data_chunks, axis=1)

        # 月初／月末価格を抽出
        open_prices  = data["Open"].iloc[0]
        close_prices = data["Adj Close"].iloc[-1]
        # 出来高のDataFrameを準備
        volume = data["Volume"]
        # 過去20日平均出来高（最終20日分の平均）
        volume_20d_avg = volume.tail(20).mean()
        # 直近5日平均出来高（最終5日分の平均）
        volume_5d_avg = volume.tail(5).mean()
        # 出来高増加率（5日平均 ÷ 20日平均）
        volume_increase_ratio = (volume_5d_avg / volume_20d_avg)*100
        
        # 騰落計算
        result = pd.DataFrame({
            "ticker":      open_prices.index,
            "start_price": open_prices.values,
            "end_price":   close_prices.values,
        })
        result["code"]        = result["ticker"].str.replace(r"\.T$", "", regex=True)
        result["price_diff"]  = result["end_price"] - result["start_price"]
        result["pct_change"]  = (result["price_diff"] / result["start_price"]) * 100
        result["volume_increase_ratio"] = result["ticker"].map(volume_increase_ratio)

        # 上位 top_n を抽出・整形
        output = (
            result
            .sort_values("pct_change", ascending=False)
            .head(top_n)
            .loc[:, ["code", "start_price", "end_price", "price_diff", "pct_change","volume_increase_ratio"]]
            .reset_index(drop=True)
        )

        pd.options.display.float_format = "{:.2f}".format
        # 列名を日本語に変更するための辞書を定義
        japanese_columns = {
            "code": "銘柄",
            "start_price": "始値",
            "end_price": "終値",
            "price_diff": "値幅",
            "pct_change": "騰落率",
            "volume_increase_ratio": "出来高急増率（5日平均＞20日平均）"
        }

        # output データフレームの列名を変更
        output_renamed = output.rename(columns=japanese_columns)
        output_renamed["騰落率"] = output_renamed["騰落率"].apply(lambda x: f"{x:.2f}%")
        output_renamed["出来高急増率（5日平均＞20日平均）"] = output_renamed["出来高急増率（5日平均＞20日平均）"].apply(lambda x: f"{x:.2f}%")
        st.write(f"対象期間: {target_start_date} 〜 {target_end_date}（上位{top_n}位）\n")
        st.dataframe(output_renamed,hide_index=True)
    
