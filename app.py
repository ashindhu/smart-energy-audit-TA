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
# ==========================
# THINGSPEAK
# ==========================

CHANNEL_ID = "3385312"
READ_API_KEY = "B36V7H0SQ125V51W"

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("⚡ Smart Energy Audit")

st.sidebar.markdown("---")

st.sidebar.subheader("ThingSpeak")

st.sidebar.write(f"Channel : {CHANNEL_ID}")

st.sidebar.write("Refresh : 60 detik")

st.sidebar.markdown("---")

st.sidebar.subheader("Model")

st.sidebar.success("Random Forest")

st.sidebar.success("LSTM")

st.sidebar.markdown("---")

st.sidebar.subheader("Universitas")

st.sidebar.write("Telkom University")

st.sidebar.write("Teknik Elektro")

st.title("Smart Energy Audit Dashboard")

st.markdown(
    """
### Prediksi Konsumsi Daya Gedung Menggunakan Machine Learning

Model yang digunakan:

- Random Forest Regression
- Long Short-Term Memory (LSTM)

Dashboard terhubung secara realtime dengan ThingSpeak dan melakukan forecasting konsumsi daya setiap jam.
"""
)


# Refresh otomatis setiap 60 detik
st_autorefresh(
    interval=60000,
    key="refresh"
)


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

st.subheader("Informasi Sistem")

col1, col2, col3 = st.columns(3)


with col1:
    st.metric(
        "Update Terakhir",
        hourly.index[-1].strftime("%H:%M")
    )

with col2:
    st.metric(
        "Random Forest",
        "READY"
    )

with col3:
    st.metric(
        "LSTM",
        "READY"
    )

st.caption(
    f"Channel ThingSpeak : {CHANNEL_ID} | "
    f"Update terakhir : {hourly.index[-1].strftime('%d-%m-%Y %H:%M WIB')}"
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
    # FORECAST 5 JAM KEDEPAN
    # ==========================

    st.subheader("Forecast 5 Jam Kedepan")

    future_time = pd.date_range(
        start=hourly.index[-1] + pd.Timedelta(hours=1),
        periods=5,
        freq="h"
    )

    forecast_df = pd.DataFrame({
        "Waktu": future_time,
        "Random Forest": rf_forecast,
        "LSTM": lstm_forecast
    })
    forecast_df["Waktu"] = (
    forecast_df["Waktu"]
    .dt.strftime("%d-%m %H:%M")
)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Random Forest")
        st.dataframe(
            forecast_df[["Waktu", "Random Forest"]],
            use_container_width=True
        )

    with col2:
        st.markdown("### LSTM")
        st.dataframe(
            forecast_df[["Waktu", "LSTM"]],
            use_container_width=True
        )

    plot_forecast = pd.DataFrame({
        "Waktu": [hourly.index[-1]] + list(future_time),
        "Aktual": [hourly.iloc[-1]] + [None] * 5,
        "Random Forest": [hourly.iloc[-1]] + rf_forecast,
        "LSTM": [hourly.iloc[-1]] + lstm_forecast
    })

    fig = px.line(
        plot_forecast,
        x="Waktu",
        y=["Aktual", "Random Forest", "LSTM"],
        markers=True,
        title="Perbandingan Forecast 5 Jam Kedepan"
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

    # ==========================
    # PERFORMA MODEL
    # ==========================

    st.subheader(
        "Perbandingan Performa Model"
    )

    result = pd.DataFrame({
        "Model": [
            "Random Forest",
            "LSTM"
        ],
        "MAE": [
            991.33,
            1008.68
        ],
        "RMSE": [
            1385.22,
            1320.19
        ],
        "R²": [
            0.8207,
            0.8371
        ]
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

    fig_compare = px.scatter(
        compare_df,
        x="Model",
        y="Power",
        size="Power",
        color="Model",
        title="Aktual vs Forecast"
    )

    fig_compare.update_traces(
        marker=dict(size=25)
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )
    
    # ==========================
    # ANALISIS MODEL
    # ==========================

    st.subheader("Analisis Forecast Terbaru")

    rf_error = validation.loc[0, "Error"]
    lstm_error = validation.loc[1, "Error"]


    if rf_error < lstm_error:

        st.success(
            f"""
    🏆 Model Terbaik Saat Ini : **Random Forest**

    Selisih prediksi Random Forest : **{rf_error:.2f} Watt**

    Selisih prediksi LSTM : **{lstm_error:.2f} Watt**

    Random Forest memiliki selisih prediksi yang lebih kecil terhadap data aktual sehingga memberikan hasil forecasting yang lebih akurat pada pembaruan data terbaru.
    """
        )

    else:

        st.success(
            f"""
    🏆 Model Terbaik Saat Ini : **LSTM**

    Selisih prediksi Random Forest : **{rf_error:.2f} Watt**

    Selisih prediksi LSTM : **{lstm_error:.2f} Watt**


    LSTM memiliki selisih prediksi yang lebih kecil terhadap data aktual sehingga memberikan hasil forecasting yang lebih akurat pada pembaruan data terbaru.
    """
        )

    # ==========================
    # HISTORY RANDOM FOREST
    # ==========================

    st.subheader("History Evaluasi Random Forest")

    history_rf = pd.read_csv("history_rf.csv")

    history_rf["Waktu"] = pd.to_datetime(history_rf["Waktu"])

    fig_rf = px.line(
        history_rf,
        x="Waktu",
        y=["Actual", "Prediction"],
        markers=True,
        title="Random Forest : Aktual vs Prediksi"
    )

    fig_rf.update_layout(
        hovermode="x unified",
        xaxis_title="Waktu",
        yaxis_title="Power (Watt)"
    )

    st.plotly_chart(
        fig_rf,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "MAE History RF",
            f"{history_rf['Error'].mean():.2f} W"
        )

    with col2:
        st.metric(
            "MAPE History RF",
            f"{history_rf['MAPE'].mean():.2f} %"
        )
    # ==========================
    # HISTORY LSTM
    # ==========================

    st.subheader("History Evaluasi LSTM")

    history_lstm = pd.read_csv("history_lstm.csv")

    history_lstm["Waktu"] = pd.to_datetime(history_lstm["Waktu"])

    fig_lstm = px.line(
        history_lstm,
        x="Waktu",
        y=["Actual", "Prediction"],
        markers=True,
        title="LSTM : Aktual vs Prediksi"
    )

    fig_lstm.update_layout(
        hovermode="x unified",
        xaxis_title="Waktu",
        yaxis_title="Power (Watt)"
    )

    st.plotly_chart(
        fig_lstm,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "MAE History LSTM",
            f"{history_lstm['Error'].mean():.2f} W"
        )

    with col2:
        st.metric(
            "MAPE History LSTM",
            f"{history_lstm['MAPE'].mean():.2f} %"
        )
    

else:

    st.warning(
        "Data belum cukup untuk prediksi."
    )