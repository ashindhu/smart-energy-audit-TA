import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from streamlit_autorefresh import st_autorefresh
from tensorflow.keras.models import load_model

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Smart Energy Audit",
    layout="wide"
)

st.title("Smart Energy Audit")
st.write("Forecasting Konsumsi Daya Menggunakan Random Forest dan LSTM")


# Refresh otomatis setiap 60 detik
st_autorefresh(
    interval=60000,
    key="refresh"
)

# ==========================
# THINGSPEAK
# ==========================

CHANNEL_ID = "3385312"
READ_API_KEY = "B36V7H0SQ125V51W"

# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_models():

    rf_model = joblib.load("rf_model.pkl")

    scaler = joblib.load("scaler.pkl")

    lstm_model = load_model(
        "lstm_model.keras",
        compile=False
    )

    return rf_model, scaler, lstm_model


rf_model, scaler, lstm_model = load_models()

# ==========================
# LOAD DATA
# ==========================

url = (
    f"https://api.thingspeak.com/channels/"
    f"{CHANNEL_ID}/feeds.csv"
    f"?api_key={READ_API_KEY}"
    f"&results=8000"
)

df = pd.read_csv(url)

df["created_at"] = pd.to_datetime(
    df["created_at"],
    utc=True
)

df["created_at"] = (
    df["created_at"]
    .dt.tz_convert("Asia/Jakarta")
)

df["field7"] = pd.to_numeric(
    df["field7"],
    errors="coerce"
)

df = df.dropna(subset=["field7"])

# ==========================
# RESAMPLE PER JAM
# ==========================

hourly = (
    df.set_index("created_at")["field7"]
    .resample("h")
    .mean()
)

hourly = hourly.interpolate(limit=3)
hourly = hourly.dropna()

# ==========================
# DATA REALTIME
# ==========================

st.subheader("Data Realtime Per Jam")

plot_df = pd.DataFrame({
    "Waktu": hourly.index,
    "Power": hourly.values
})

fig = px.line(
    plot_df,
    x="Waktu",
    y="Power",
    markers=True,
    title="Konsumsi Daya Aktual"
)

fig.update_layout(
    hovermode="x unified",
    xaxis_title="Waktu (WIB)",
    yaxis_title="Power (Watt)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Informasi Data")

st.write(f"Channel ID : {CHANNEL_ID}")

st.write(f"Jumlah Data Per Jam : {len(hourly)}")

st.write(
    "Update Terakhir : "
    + hourly.index[-1].strftime("%d-%m-%Y %H:%M WIB")
)

# ==========================
# FORECAST
# ==========================

if len(hourly) >= 24:

    latest = hourly.iloc[-24:]

    # ==========================
    # RANDOM FOREST 5 JAM
    # ==========================

    rf_history = list(hourly.values)

    rf_forecast = []

    for _ in range(5):

        rf_input = pd.DataFrame({
            "lag_1": [rf_history[-1]],
            "lag_2": [rf_history[-2]],
            "lag_3": [rf_history[-3]],
            "lag_6": [rf_history[-6]],
            "lag_12": [rf_history[-12]],
            "lag_24": [rf_history[-24]]
        })

        pred = rf_model.predict(
            rf_input
        )[0]

        rf_forecast.append(pred)

        rf_history.append(pred)

    # ==========================
    # LSTM 5 JAM
    # ==========================

    lstm_window = scaler.transform(
        latest.values.reshape(-1, 1)
    )

    current_window = lstm_window.copy()

    lstm_forecast = []

    for _ in range(5):

        X_lstm = current_window.reshape(
            1,
            24,
            1
        )

        pred = lstm_model.predict(
            X_lstm,
            verbose=0
        )

        pred_real = scaler.inverse_transform(
            pred
        )[0][0]

        lstm_forecast.append(
            pred_real
        )

        current_window = np.vstack([
            current_window[1:],
            pred
        ])

    # ==========================
    # METRIC CARD
    # ==========================

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "RF Prediksi +1 Jam",
            f"{rf_forecast[0]:.2f} W"
        )

    with col2:
        st.metric(
            "LSTM Prediksi +1 Jam",
            f"{lstm_forecast[0]:.2f} W"
        )

    # ==========================
    # RF FORECAST
    # ==========================

    st.subheader(
        "Forecasting Random Forest (5 Jam Kedepan)"
    )

    rf_df = pd.DataFrame({
        "Jam": ["+1", "+2", "+3", "+4", "+5"],
        "Power": rf_forecast
    })

    st.dataframe(
        rf_df,
        use_container_width=True
    )

    fig_rf = px.line(
        rf_df,
        x="Jam",
        y="Power",
        markers=True,
        title="Forecast Random Forest"
    )

    fig_rf.update_layout(
        xaxis_title="Jam Kedepan",
        yaxis_title="Power (Watt)"
    )

    st.plotly_chart(
        fig_rf,
        use_container_width=True
    )

    # ==========================
    # LSTM FORECAST
    # ==========================

    st.subheader(
        "Forecasting LSTM (5 Jam Kedepan)"
    )

    lstm_df = pd.DataFrame({
        "Jam": ["+1", "+2", "+3", "+4", "+5"],
        "Power": lstm_forecast
    })

    st.dataframe(
        lstm_df,
        use_container_width=True
    )

    fig_lstm = px.line(
        lstm_df,
        x="Jam",
        y="Power",
        markers=True,
        title="Forecast LSTM"
    )

    fig_lstm.update_layout(
        xaxis_title="Jam Kedepan",
        yaxis_title="Power (Watt)"
    )

    st.plotly_chart(
        fig_lstm,
        use_container_width=True
    )

    # ==========================
    # PERFORMA MODEL
    # ==========================

    st.subheader(
        "Perbandingan Performa Model"
    )

    result = pd.DataFrame({
        "Model": ["Random Forest", "LSTM"],
        "MAE": [1001.84, 1008.83],
        "RMSE": [1398.88, 1363.49],
        "R²": [0.8490, 0.8565]
    })

    st.dataframe(
        result,
        use_container_width=True
    )

    # ==========================
    # VALIDASI
    # ==========================

    st.subheader(
        "Perbandingan Forecast Dengan Data Aktual"
    )

    actual = hourly.iloc[-1]

    validation = pd.DataFrame({
        "Model": [
            "Random Forest",
            "LSTM"
        ],
        "Forecast": [
            rf_forecast[0],
            lstm_forecast[0]
        ],
        "Aktual Terakhir": [
            actual,
            actual
        ]
    })

    validation["Error"] = (
        validation["Aktual Terakhir"]
        - validation["Forecast"]
    ).abs()

    validation["MAPE (%)"] = (
        validation["Error"]
        / validation["Aktual Terakhir"]
        * 100
    )

    st.dataframe(
        validation,
        use_container_width=True
    )

    compare_df = pd.DataFrame({
        "Model": [
            "Aktual",
            "Random Forest",
            "LSTM"
        ],
        "Power": [
            actual,
            rf_forecast[0],
            lstm_forecast[0]
        ]
    })

    fig_compare = px.bar(
        compare_df,
        x="Model",
        y="Power",
        title="Aktual vs Forecast"
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

else:

    st.warning(
        "Data belum cukup untuk prediksi."
    )