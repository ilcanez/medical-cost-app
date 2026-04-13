import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =============================================
# LOAD MODEL & SCALER
# =============================================
model  = joblib.load('model_xgboost.pkl')
scaler = joblib.load('scaler.pkl')

# =============================================
# KONFIGURASI HALAMAN
# =============================================
st.set_page_config(page_title="Prediksi Biaya Medis", page_icon="🏥", layout="centered")

st.title("🏥 Prediksi Biaya Medis Tahunan")
st.markdown("Isi data pasien di bawah ini untuk memprediksi estimasi biaya medis tahunan.")
st.divider()

# =============================================
# FORM INPUT
# =============================================
st.subheader("👤 Data Demografi")
col1, col2 = st.columns(2)

with col1:
    age    = st.number_input("Usia", min_value=0, max_value=100, value=40)
    sex    = st.selectbox("Jenis Kelamin", ["Female", "Male", "Other"])
    region = st.selectbox("Wilayah", ["North", "Central", "West", "South", "East"])
    urban_rural = st.selectbox("Tipe Area", ["Urban", "Suburban", "Rural"])

with col2:
    income         = st.number_input("Pendapatan Tahunan (Rp)", min_value=0, value=50000)
    education      = st.selectbox("Pendidikan", ["No HS", "HS", "Some College", "Bachelors", "Masters", "Doctorate"])
    marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced", "Widowed"])
    employment_status = st.selectbox("Status Pekerjaan", ["Employed", "Unemployed", "Self-employed", "Retired"])

st.divider()
st.subheader("🏠 Data Rumah Tangga")
col3, col4 = st.columns(2)
with col3:
    household_size = st.number_input("Jumlah Anggota Keluarga", min_value=1, max_value=20, value=3)
with col4:
    dependents = st.number_input("Jumlah Tanggungan", min_value=0, max_value=20, value=1)

st.divider()
st.subheader("🩺 Data Kesehatan")
col5, col6 = st.columns(2)

with col5:
    bmi         = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    smoker      = st.selectbox("Status Merokok", ["Never", "Former", "Current"])
    alcohol_freq = st.selectbox("Frekuensi Alkohol", ["Unknown", "Occasional", "Weekly", "Daily"])
    systolic_bp  = st.number_input("Tekanan Darah Sistolik", min_value=80, max_value=200, value=120)
    diastolic_bp = st.number_input("Tekanan Darah Diastolik", min_value=50, max_value=130, value=80)

with col6:
    ldl              = st.number_input("LDL Kolesterol", min_value=50, max_value=300, value=120)
    hba1c            = st.number_input("HbA1c", min_value=4.0, max_value=15.0, value=5.5)
    medication_count = st.number_input("Jumlah Obat Rutin", min_value=0, max_value=20, value=0)
    risk_score       = st.slider("Risk Score", min_value=0.0, max_value=1.0, value=0.5)

st.divider()
st.subheader("🏨 Riwayat Medis")
col7, col8 = st.columns(2)
with col7:
    visits_last_year            = st.number_input("Kunjungan Dokter (setahun terakhir)", min_value=0, max_value=50, value=2)
    hospitalizations_last_3yrs  = st.number_input("Rawat Inap (3 tahun terakhir)", min_value=0, max_value=20, value=0)
    days_hospitalized_last_3yrs = st.number_input("Total Hari Rawat Inap", min_value=0, max_value=365, value=0)
    claims_count                = st.number_input("Jumlah Klaim", min_value=0, max_value=30, value=1)

with col8:
    st.markdown("**Penyakit Kronis:**")
    hypertension           = int(st.checkbox("Hipertensi"))
    diabetes               = int(st.checkbox("Diabetes"))
    asthma                 = int(st.checkbox("Asma"))
    copd                   = int(st.checkbox("COPD"))
    cardiovascular_disease = int(st.checkbox("Penyakit Jantung"))
    cancer_history         = int(st.checkbox("Riwayat Kanker"))
    kidney_disease         = int(st.checkbox("Penyakit Ginjal"))
    liver_disease          = int(st.checkbox("Penyakit Hati"))
    arthritis              = int(st.checkbox("Arthritis"))
    mental_health          = int(st.checkbox("Masalah Kesehatan Mental"))

chronic_count = sum([hypertension, diabetes, asthma, copd,
                     cardiovascular_disease, cancer_history,
                     kidney_disease, liver_disease, arthritis, mental_health])

st.divider()
st.subheader("📋 Data Asuransi & Prosedur")
col9, col10 = st.columns(2)
with col9:
    plan_type             = st.selectbox("Tipe Plan", ["HMO", "PPO", "EPO", "POS"])
    network_tier          = st.selectbox("Network Tier", ["Bronze", "Silver", "Gold", "Platinum"])
    deductible            = st.number_input("Deductible", min_value=0, value=500)
    copay                 = st.number_input("Copay", min_value=0, value=30)
    policy_term_years     = st.number_input("Lama Polis (tahun)", min_value=1, max_value=30, value=5)
    policy_changes_last_2yrs = st.number_input("Perubahan Polis (2 tahun terakhir)", min_value=0, max_value=10, value=0)
    provider_quality      = st.slider("Kualitas Provider (1-5)", min_value=1, max_value=5, value=3)

with col10:
    st.markdown("**Jumlah Prosedur Medis:**")
    proc_imaging_count = st.number_input("Imaging (Rontgen/MRI)", min_value=0, max_value=20, value=0)
    proc_surgery_count = st.number_input("Operasi", min_value=0, max_value=10, value=0)
    proc_physio_count  = st.number_input("Fisioterapi", min_value=0, max_value=20, value=0)
    proc_consult_count = st.number_input("Konsultasi", min_value=0, max_value=30, value=1)
    proc_lab_count     = st.number_input("Lab/Tes Darah", min_value=0, max_value=20, value=1)

is_high_risk      = int(st.checkbox("Pasien High Risk"))
had_major_procedure = int(st.checkbox("Pernah Prosedur Besar"))

st.divider()

# =============================================
# PREPROCESSING INPUT
# =============================================
def preprocess_input():
    # Feature Engineering
    total_procedures = proc_imaging_count + proc_surgery_count + proc_physio_count + proc_consult_count + proc_lab_count
    bmi_age_index    = bmi * age
    smoker_resp_risk = int(smoker == 'Current' and (asthma == 1 or copd == 1))

    # Ordinal Encoding
    education_map  = {'No HS': 0, 'HS': 1, 'Some College': 2, 'Bachelors': 3, 'Masters': 4, 'Doctorate': 5}
    smoker_map     = {'Never': 0, 'Former': 1, 'Current': 2}
    alcohol_map    = {'Unknown': 0, 'Occasional': 1, 'Weekly': 2, 'Daily': 3}
    tier_map       = {'Bronze': 0, 'Silver': 1, 'Gold': 2, 'Platinum': 3}

    # Log transform income
    income_log = np.log1p(income)

    # Base numerical features
    data = {
        'age': age, 'income': income_log,
        'household_size': household_size, 'dependents': dependents,
        'bmi': bmi, 'visits_last_year': visits_last_year,
        'hospitalizations_last_3yrs': hospitalizations_last_3yrs,
        'days_hospitalized_last_3yrs': days_hospitalized_last_3yrs,
        'medication_count': medication_count,
        'systolic_bp': systolic_bp, 'diastolic_bp': diastolic_bp,
        'ldl': ldl, 'hba1c': hba1c,
        'deductible': deductible, 'copay': copay,
        'policy_term_years': policy_term_years,
        'policy_changes_last_2yrs': policy_changes_last_2yrs,
        'provider_quality': provider_quality,
        'risk_score': risk_score, 'claims_count': claims_count,
        'chronic_count': chronic_count,
        'hypertension': hypertension, 'diabetes': diabetes,
        'asthma': asthma, 'copd': copd,
        'cardiovascular_disease': cardiovascular_disease,
        'cancer_history': cancer_history, 'kidney_disease': kidney_disease,
        'liver_disease': liver_disease, 'arthritis': arthritis,
        'mental_health': mental_health,
        'proc_imaging_count': proc_imaging_count,
        'proc_surgery_count': proc_surgery_count,
        'proc_physio_count': proc_physio_count,
        'proc_consult_count': proc_consult_count,
        'proc_lab_count': proc_lab_count,
        'is_high_risk': is_high_risk,
        'had_major_procedure': had_major_procedure,
        'education': education_map[education],
        'smoker': smoker_map[smoker],
        'alcohol_freq': alcohol_map[alcohol_freq],
        'network_tier': tier_map[network_tier],
        'total_procedures': total_procedures,
        'bmi_age_index': bmi_age_index,
        'smoker_resp_risk': smoker_resp_risk,
        # OHE: sex
        'sex_Male': int(sex == 'Male'),
        'sex_Other': int(sex == 'Other'),
        # OHE: region
        'region_East': int(region == 'East'),
        'region_North': int(region == 'North'),
        'region_South': int(region == 'South'),
        'region_West': int(region == 'West'),
        # OHE: urban_rural
        'urban_rural_Suburban': int(urban_rural == 'Suburban'),
        'urban_rural_Urban': int(urban_rural == 'Urban'),
        # OHE: marital_status
        'marital_status_Married': int(marital_status == 'Married'),
        'marital_status_Single': int(marital_status == 'Single'),
        'marital_status_Widowed': int(marital_status == 'Widowed'),
        # OHE: employment_status
        'employment_status_Retired': int(employment_status == 'Retired'),
        'employment_status_Self-employed': int(employment_status == 'Self-employed'),
        'employment_status_Unemployed': int(employment_status == 'Unemployed'),
        # OHE: plan_type
        'plan_type_HMO': int(plan_type == 'HMO'),
        'plan_type_POS': int(plan_type == 'POS'),
        'plan_type_PPO': int(plan_type == 'PPO'),
    }

    df_input = pd.DataFrame([data])

    # Scale kolom kontinu
    cols_to_scale = [
        'age', 'income', 'education', 'household_size', 'dependents',
        'bmi', 'smoker', 'alcohol_freq', 'visits_last_year',
        'hospitalizations_last_3yrs', 'days_hospitalized_last_3yrs',
        'medication_count', 'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c',
        'network_tier', 'deductible', 'copay', 'policy_term_years',
        'policy_changes_last_2yrs', 'provider_quality', 'risk_score',
        'claims_count', 'chronic_count', 'proc_imaging_count',
        'proc_surgery_count', 'proc_physio_count', 'proc_consult_count',
        'proc_lab_count', 'total_procedures', 'bmi_age_index'
    ]

    # Reorder kolom sesuai urutan saat training
    expected_cols = [
        'age', 'income', 'education', 'household_size', 'dependents',
        'bmi', 'smoker', 'alcohol_freq', 'visits_last_year',
        'hospitalizations_last_3yrs', 'days_hospitalized_last_3yrs',
        'medication_count', 'systolic_bp', 'diastolic_bp', 'ldl', 'hba1c',
        'network_tier', 'deductible', 'copay', 'policy_term_years',
        'policy_changes_last_2yrs', 'provider_quality', 'risk_score',
        'claims_count', 'chronic_count', 'hypertension', 'diabetes',
        'asthma', 'copd', 'cardiovascular_disease', 'cancer_history',
        'kidney_disease', 'liver_disease', 'arthritis', 'mental_health',
        'proc_imaging_count', 'proc_surgery_count', 'proc_physio_count',
        'proc_consult_count', 'proc_lab_count', 'is_high_risk',
        'had_major_procedure', 'total_procedures', 'bmi_age_index',
        'smoker_resp_risk', 'sex_Male', 'sex_Other',
        'region_East', 'region_North', 'region_South', 'region_West',
        'urban_rural_Suburban', 'urban_rural_Urban',
        'marital_status_Married', 'marital_status_Single', 'marital_status_Widowed',
        'employment_status_Retired', 'employment_status_Self-employed',
        'employment_status_Unemployed',
        'plan_type_HMO', 'plan_type_POS', 'plan_type_PPO'
    ]

    df_input = df_input[expected_cols]
    df_input[cols_to_scale] = scaler.transform(df_input[cols_to_scale])
    return df_input

# =============================================
# PREDIKSI
# =============================================
if st.button("🔍 Prediksi Biaya Medis", type="primary", use_container_width=True):
    try:
        input_data   = preprocess_input()
        pred_log     = model.predict(input_data)[0]
        pred_actual  = np.expm1(pred_log)  # balik dari log

        st.divider()
        st.subheader("📊 Hasil Prediksi")

        col_res1, col_res2, col_res3 = st.columns(3)
        with col_res1:
            st.metric("Estimasi Biaya Medis Tahunan", f"${pred_actual:,.2f}")
        with col_res2:
            st.metric("Jumlah Penyakit Kronis", chronic_count)
        with col_res3:
            risk_label = "🔴 Tinggi" if risk_score > 0.7 else "🟡 Sedang" if risk_score > 0.4 else "🟢 Rendah"
            st.metric("Level Risiko", risk_label)

        st.info("ℹ️ Prediksi ini bersifat estimasi berdasarkan model machine learning. "
                "Untuk keputusan medis, selalu konsultasikan dengan profesional kesehatan.")

    except Exception as e:
        st.error(f"Terjadi error: {e}")
