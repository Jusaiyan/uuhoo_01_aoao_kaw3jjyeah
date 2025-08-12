import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from dateutil.relativedelta import relativedelta
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler

def run_stock_forecast_app():
    st.set_page_config(page_title="銘柄チャート予測（1か月）", layout="wide")

    st.title("チャート予測 — 過去データから1か月先を予測")
    st.caption("yfinanceで価格取得し、TensorFlowの軽量LSTMモデルで予測します。")

    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        ticker = st.text_input("銘柄（例: 7203.T または 6758.T）", value="285A.T")
    with col2:
        hist_months = st.number_input("取得する過去データ（月）", min_value=1, max_value=6, value=3)
    with col3:
        forecast_days = st.number_input("予測営業日数", min_value=5, max_value=30, value=22)

    st.sidebar.header("LSTM ハイパーパラメータ")
    seq_len = st.sidebar.number_input("シーケンス長（日数）", min_value=5, max_value=30, value=20)
    epochs = st.sidebar.number_input("学習エポック数", min_value=5, max_value=50, value=20)
    batch_size = st.sidebar.number_input("バッチサイズ", min_value=1, max_value=32, value=8)

    run = st.button("実行 ▶︎ 予測開始")
    if not run:
        st.info("上記の条件を設定後、[実行] を押してください。")
        return

    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
    except Exception:
        st.error("TensorFlowがインストールされていません。requirements.txtに'tensorflow'を追加してください。")
        return

    end_date = datetime.date.today()
    start_date = end_date - relativedelta(months=hist_months)

    with st.spinner(f"{ticker} のデータを yfinance から取得中..."):
        df = yf.download(tickers=ticker, start=str(start_date), end=str(end_date), progress=False, auto_adjust=True)
    if df.empty:
        st.error("データが取得できませんでした。ティッカーと期間を確認してください。")
        return

    price = df['Close'].dropna().to_frame()

    st.subheader(f"取得データ: {ticker} ({start_date}〜{end_date}) {len(price)}日分")
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=price.index, y=price['Close'], name='実績終値'))
    fig_hist.update_layout(title='実績終値', xaxis_title='日付', yaxis_title='価格')
    st.plotly_chart(fig_hist, use_container_width=True)

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(price.values)

    if len(scaled_values) <= seq_len + 1:
        st.error("過去データが短すぎて学習できません。取得期間を増やしてください。")
        return

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data[i+seq_length])
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_values, seq_len)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = Sequential([
        LSTM(32, input_shape=(seq_len, 1)),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    with st.spinner(f"LSTMモデルを {epochs} エポック学習中..."):
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)

    st.success("学習完了")

    last_seq = scaled_values[-seq_len:].reshape((1, seq_len, 1))
    preds_scaled = []
    for _ in range(forecast_days):
        pred = model.predict(last_seq, verbose=0)[0, 0]
        preds_scaled.append(pred)
        last_seq = np.roll(last_seq, -1)
        last_seq[0, -1, 0] = pred

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    forecast_index = pd.bdate_range(start=price.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    df_forecast = pd.DataFrame({'Forecast': preds}, index=forecast_index)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price.index, y=price['Close'], name='実績終値'))
    fig.add_trace(go.Scatter(x=df_forecast.index, y=df_forecast['Forecast'], name='予測終値（LSTM）'))
    fig.update_layout(title=f"{ticker} のLSTM予測 ({forecast_days}営業日)", xaxis_title='日付', yaxis_title='価格')
    st.plotly_chart(fig, use_container_width=True)

    st.info("※本予測は参考値です。投資判断は自己責任でお願いします。")

#if __name__ == "__main__":
#    run_stock_forecast_app()
