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

st.sidebar.subheader("Universitas")

st.sidebar.write("Telkom University")

st.sidebar.write("Teknik Elektro")
st.sidebar.markdown("---")


st.title("Smart Energy Audit Dashboard")

st.markdown(
    """
### Monitoring Konsumsi Energi Gedung Berbasis Machine Learning

Dashboard ini digunakan untuk melakukan monitoring konsumsi energi secara realtime, analisis baseline audit energi, serta forecasting konsumsi daya menggunakan Random Forest Regression dan Long Short-Term Memory (LSTM).

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
# LOAD PROFILE
# ==========================

current_power = hourly.iloc[-1]

peak_power = hourly.max()
peak_time = hourly.idxmax()

offpeak_power = hourly.min()
offpeak_time = hourly.idxmin()

print("Peak Time :", peak_time)
print("Off Peak Time :", offpeak_time)

print("Peak Power :", peak_power)
print("Off Peak Power :", offpeak_power)

# ==========================
# BASELINE RANDOM FOREST
# ==========================

latest = hourly.iloc[-24:]

rf_history = list(hourly.values)

rf_input = pd.DataFrame({
    "lag_1": [rf_history[-1]],
    "lag_2": [rf_history[-2]],
    "lag_3": [rf_history[-3]],
    "lag_6": [rf_history[-6]],
    "lag_12": [rf_history[-12]],
    "lag_24": [rf_history[-24]]
})

baseline_rf = rf_model.predict(rf_input)[0]

# ==========================
# DATA REALTIME
# ==========================

st.subheader("Data Realtime Per Jam")

plot_df = pd.DataFrame({
    "Waktu": hourly.index,
    "Power": hourly.values / 1000
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
    yaxis_title="Power (kW)"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Informasi Sistem")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Jumlah Data", len(hourly))

with col2:
    st.metric("Update Terakhir",
              hourly.index[-1].strftime("%H:%M"))

with col3:
    st.metric("Model Baseline",
              "Random Forest")

st.caption(
    f"Channel ThingSpeak : {CHANNEL_ID} | "
    f"Update terakhir : {hourly.index[-1].strftime('%d-%m-%Y %H:%M WIB')}"
)


    # ==========================
    # HASIL AUDIT ENERGI
    # ==========================

st.header("📋 Hasil Audit Energi")

current_power = hourly.iloc[-1]

current_power_kw = current_power / 1000

baseline_kw = baseline_rf / 1000

delta = (
    (current_power - baseline_rf)
    / baseline_rf
) * 100

if delta > 15:

    status = "🔴 Boros"

elif delta > 5:

    status = "🟡 Waspada"

elif delta >= -5:

    status = "🟢 Normal"

else:

    status = "🟢 Hemat"

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Konsumsi Aktual",
        f"{current_power_kw:.2f} kW"
    )

with col2:
    st.metric(
        "Baseline RF",
        f"{baseline_kw:.2f} kW"
    )

with col3:
    st.metric(
        "Deviasi Δ",
        f"{delta:.2f}%"
    )

col4, col5, col6, col7 = st.columns(4)

with col4:
    st.metric(
    "Peak Load",
    f"{peak_power/1000:.2f} kW",
    peak_time.strftime("%H:%M WIB")
)

with col5:
    st.metric(
    "Off Peak",
    f"{offpeak_power/1000:.2f} kW",
    offpeak_time.strftime("%H:%M WIB")
)

with col6:
    st.metric(
        "Status Audit",
        status
    )


# ==========================
# PROFIL BEBAN HARIAN
# ==========================

st.subheader("Profil Beban Harian")

daily_profile = (
    hourly.groupby(hourly.index.hour)
    .mean()
    .reset_index()
)

daily_profile.columns = [
    "Jam",
    "Power"
]
daily_profile["Power"] = daily_profile["Power"] / 1000
daily_profile["Jam_Label"] = (
    daily_profile["Jam"]
    .apply(lambda x: f"{x:02d}.00")
)


fig_daily = px.line(
    daily_profile,
    x="Jam_Label",
    y="Power",
    markers=True,
    title="Rata-rata Konsumsi Daya per Jam"
)

fig_daily.update_layout(
    xaxis_title="Jam",
    yaxis_title="Power (kW)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_daily,
    use_container_width=True
)

# ==========================
# HASIL ANALISIS AUDIT
# ==========================

st.subheader("Hasil Analisis Audit Energi")

peak_value = hourly.max()
offpeak_value = hourly.min()

st.info(f"""
### Ringkasan Audit Energi

• Konsumsi Aktual : **{current_power_kw:.2f} kW**

• Baseline Random Forest : **{baseline_kw:.2f} kW**

• Deviasi Load Profiling : **{delta:.2f}%**

• Status Audit : **{status}**

### Interpretasi
""")

if delta > 15:
    st.error("Deviasi di atas 15%. Terindikasi terjadi pemborosan energi sehingga diperlukan evaluasi beban dan peralatan listrik.")

elif delta > 5:
    st.warning("Deviasi berada pada rentang 5–15%. Konsumsi energi mulai menyimpang dari baseline sehingga perlu dilakukan pemantauan.")

elif delta >= -5:
    st.success("Konsumsi energi masih berada di sekitar baseline sehingga kondisi operasional dinilai normal.")

else:
    st.success("Konsumsi energi berada di bawah baseline sehingga penggunaan energi lebih efisien dibanding kondisi normal.")

### Rekomendasi

if delta > 15:

    st.error("""
• Lakukan inspeksi peralatan dengan konsumsi tinggi.

• Evaluasi jadwal operasi HVAC dan pencahayaan.

• Periksa kemungkinan beban yang tidak diperlukan.
""")

elif delta > 5:

    st.warning("""
• Lakukan monitoring konsumsi energi.

• Evaluasi perubahan pola beban.
""")

else:

    st.success("""
• Konsumsi energi masih sesuai baseline.

• Tidak diperlukan tindakan korektif.
""")

# ==========================
# BASELINE PENILAIAN
# ==========================

with st.expander("📖 Baseline Penilaian Audit"):

    st.markdown("""
### Baseline Penilaian Audit

Pada penelitian ini, proses audit energi menggunakan pendekatan **Load Profiling**, yaitu membandingkan konsumsi daya aktual dengan **baseline** yang dihasilkan oleh model **Random Forest Regression**.

Nilai deviasi dihitung menggunakan persamaan:

Δ = (Aktual − Baseline) / Baseline × 100%

Interpretasi hasil audit yang digunakan pada penelitian ini adalah:

- 🟢 **Normal** : Deviasi ≤ 5%
- 🟡 **Waspada** : Deviasi > 5% hingga 15%
- 🔴 **Boros** : Deviasi > 15%

Semakin kecil nilai deviasi, maka pola konsumsi energi semakin mendekati kondisi normal sehingga operasi gedung dinilai lebih efisien. Sebaliknya, deviasi yang tinggi mengindikasikan adanya penyimpangan pola konsumsi energi yang perlu dianalisis lebih lanjut sebagai dasar penyusunan rekomendasi efisiensi operasional.
""")



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
            f"{rf_forecast[0]/1000:.2f} kW"
        )

    with col2:
        st.metric(
            "LSTM Prediksi +1 Jam",
            f"{lstm_forecast[0]/1000:.2f} kW"
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
    forecast_df["Random Forest"] = forecast_df["Random Forest"] / 1000
    forecast_df["LSTM"] = forecast_df["LSTM"] / 1000

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
        "Aktual": [hourly.iloc[-1]/1000] + [None]*5,
        "Random Forest": [hourly.iloc[-1]/1000] + [x/1000 for x in rf_forecast],
        "LSTM": [hourly.iloc[-1]/1000] + [x/1000 for x in lstm_forecast]
    })

    fig = px.line(
        plot_forecast,
        x="Waktu",
        y=["Aktual", "Random Forest", "LSTM"],
        markers=True,
        title="Perbandingan Forecast 5 Jam Kedepan"
    )
    fig.update_traces(line=dict(width=3))

    fig.for_each_trace(
lambda t: t.update(
    line=dict(
        color={
            "Aktual": "#1f77b4",
            "Random Forest": "#2ca02c",
            "LSTM": "#d62728"
        }[t.name]
    )
)
)

    fig.update_layout(
        hovermode="x unified",
        xaxis_title="Waktu (WIB)",
        yaxis_title="Power (kW)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ==========================
    # PERFORMA MODEL
    # ==========================

    st.subheader("Evaluasi Model Machine Learning")

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

    Selisih prediksi Random Forest : {rf_error/1000:.2f} kW

    Selisih prediksi LSTM : {lstm_error/1000:.2f} kW

    Random Forest memiliki selisih prediksi yang lebih kecil terhadap data aktual sehingga memberikan hasil forecasting yang lebih akurat pada pembaruan data terbaru.
    """
        )

    else:

        st.success(
            f"""
    🏆 Model Terbaik Saat Ini : **LSTM**

    Selisih prediksi Random Forest : {rf_error/1000:.2f} kW

Selisih prediksi LSTM : {lstm_error/1000:.2f} kW

    LSTM memiliki selisih prediksi yang lebih kecil terhadap data aktual sehingga memberikan hasil forecasting yang lebih akurat pada pembaruan data terbaru.
    """
        )

    # ==========================
    # HISTORY RANDOM FOREST
    # ==========================

    st.subheader("History Evaluasi Random Forest")

    history_rf = pd.read_csv("history_rf.csv")

    history_rf["Actual"] = history_rf["Actual"] / 1000
    history_rf["Prediction"] = history_rf["Prediction"] / 1000
    history_rf["Error"] = history_rf["Error"] / 1000

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
        yaxis_title="Power (kW)"
    )

    st.plotly_chart(
        fig_rf,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "MAE History RF",
            f"{history_rf['Error'].mean():.2f} kW"
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

    history_lstm["Actual"] = history_lstm["Actual"] / 1000
    history_lstm["Prediction"] = history_lstm["Prediction"] / 1000
    history_lstm["Error"] = history_lstm["Error"] / 1000

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
        yaxis_title="Power (kW)"
    )

    st.plotly_chart(
        fig_lstm,
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "MAE History LSTM",
            f"{history_lstm['Error'].mean():.2f} kW"
        )

    with col2:
        st.metric(
            "MAPE History LSTM",
            f"{history_lstm['MAPE'].mean():.2f} %"
        )
    
    # ==========================
    # GABUNGAN HISTORY RF & LSTM
    # ==========================

    st.subheader("Perbandingan Aktual, Random Forest, dan LSTM")

    compare_history = history_rf.copy()

    compare_history["RF"] = history_rf["Prediction"]
    compare_history["LSTM"] = history_lstm["Prediction"]
    compare_history["Aktual"] = history_rf["Actual"]

    fig_compare = px.line(
        compare_history,
        x="Waktu",
        y=["Aktual", "RF", "LSTM"],
        markers=True,
        title="Perbandingan Data Aktual dengan Prediksi Random Forest dan LSTM"
    )

    # warna manual
    fig_compare.for_each_trace(
        lambda t: t.update(
            line=dict(
                width=3,
                color={
                    "Aktual": "#1f77b4",
                    "RF": "#2ca02c",
                    "LSTM": "#d62728"
                }[t.name]
            )
        )
    )

    fig_compare.update_layout(
        hovermode="x unified",
        xaxis_title="Waktu",
        yaxis_title="Power (kW)"
    )

    st.plotly_chart(
        fig_compare,
        use_container_width=True
    )

else:

    st.warning(
        "Data belum cukup untuk prediksi."
    )
    