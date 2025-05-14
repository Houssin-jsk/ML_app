import streamlit as st
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

model_path = "https://drive.google.com/file/d/1Iu5S6LzuMr9ePxy0QiY2geO_yohAILsa/view?usp=sharing"

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        try:
            data = pickle.load(f)
            if isinstance(data, dict) and "model" in data and "features" in data:
                model = data["model"]
                expected_cols = data["features"]
            else:
                st.error("Le fichier pickle ne contient pas les clés attendues ('model' et 'features').")
                st.stop()
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")
            st.stop()
else:
    st.error(f"Le fichier {model_path} est introuvable.")
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv("diabetes_prediction_dataset.csv")

df = load_data()

df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop("diabetes", axis=1)
y = df_encoded["diabetes"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Modèle Enregistré": model,
    "Random Forest": RandomForestClassifier(random_state=42)
}

tab1, tab2 = st.tabs(["🧪 Prédiction", "📊 Visualisations EDA"])

with tab1:
    st.title("🔬 Prédiction du diabète")
    st.subheader("Entrez les données du patient")

    age = st.number_input("Âge", 1, 100)
    bmi = st.number_input("IMC", 10.0, 60.0)
    glucose = st.number_input("Niveau de glucose", 50, 300)
    hba1c = st.number_input("HbA1c (%)", 3.0, 15.0)
    hypertension = st.selectbox("Hypertension", [0, 1])
    heart_disease = st.selectbox("Maladie cardiaque", [0, 1])
    gender = st.selectbox("Sexe", ["Male", "Female", "Other"])
    smoking = st.selectbox("Historique de tabagisme", ["never", "current", "former", "not current", "ever", "No Info"])

    # Encodage One-Hot
    input_dict = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose,
        'gender_Male': 0,
        'gender_Other': 0,
        'smoking_history_current': 0,
        'smoking_history_ever': 0,
        'smoking_history_former': 0,
        'smoking_history_never': 0,
        'smoking_history_not current': 0,
        'smoking_history_No Info': 0
    }

    if gender == "Male":
        input_dict['gender_Male'] = 1
    elif gender == "Other":
        input_dict['gender_Other'] = 1

    smoking_key = f"smoking_history_{smoking}"
    if smoking_key in input_dict:
        input_dict[smoking_key] = 1

    input_data = pd.DataFrame([input_dict])

    try:
        input_data = input_data[expected_cols]
    except Exception as e:
        st.error("⚠️ Les colonnes de l'entrée ne correspondent pas au modèle.")
        st.write("Colonnes attendues:", expected_cols)
        st.write("Colonnes fournies:", input_data.columns.tolist())
        st.stop()

    st.write("📊 Données entrées:", input_data)

    # Prédiction
    if st.button("Prédire"):
        prediction = model.predict(input_data)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_data)[0][1]
        else:
            proba = float(prediction[0])

        label = '🔵 Diabétique' if prediction[0] == 1 else '🔴 Non diabétique'

        st.markdown("### 🔍 Résultat de la prédiction")
        st.markdown(f"**🧬 Probabilité de diabète : `{proba*100:.2f}%`**")
        st.markdown(f"**📜 Interprétation : `{label}`**")

with tab2:
    st.title("📊 Visualisations Exploratoires")

    st.markdown("Analyse visuelle des données du dataset pour mieux comprendre les corrélations et les tendances.")

    st.subheader("🔢 Répartition des classes")
    
    fig1, ax1 = plt.subplots()
    sns.countplot(x='diabetes', data=df, palette='Set2', ax=ax1)
    ax1.set_title('Répartition des classes (Diabète)')
    ax1.set_xlabel('Diabète (0 = Non, 1 = Oui)')
    ax1.set_ylabel('Nombre')
    st.pyplot(fig1)

    st.subheader("📈 Distribution de l’âge selon le statut du diabète")
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df, x='age', hue='diabetes', bins=30, kde=True, multiple='stack', ax=ax2)
    ax2.set_title("Distribution de l'âge selon le diabète")
    ax2.set_xlabel('Âge')
    ax2.set_ylabel('Fréquence')
    st.pyplot(fig2)

    st.subheader("📈 IMC vs Niveau de glucose")
    
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='bmi', y='blood_glucose_level', hue='diabetes', palette='coolwarm', ax=ax3)
    ax3.set_title("IMC vs Niveau de glucose")
    ax3.set_xlabel('IMC')
    ax3.set_ylabel('Niveau de glucose')
    st.pyplot(fig3)

    st.subheader("📦 BMI selon l'historique de tabagisme")
    
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='smoking_history', y='bmi', hue='diabetes', ax=ax4)
    ax4.set_title("IMC et tabagisme selon le diabète")
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    st.pyplot(fig4)

    fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
    corr_matrix = df_encoded.corr()
  
    st.subheader("📊 Matrice de corrélation")
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        annot=True,           
        fmt=".2f",            
        vmin=-1, vmax=1,     
        ax=ax_corr
    )

    ax_corr.set_title("Carte de corrélation")
    st.pyplot(fig_corr)


    st.subheader("🧬 Distribution de HbA1c")
    
    fig5, ax5 = plt.subplots()
    sns.violinplot(data=df, x='diabetes', y='HbA1c_level', palette='muted', ax=ax5)
    ax5.set_title("Distribution de HbA1c selon le statut diabétique")
    ax5.set_xlabel('Diabète')
    ax5.set_ylabel('HbA1c (%)')
    st.pyplot(fig5)

    
    st.subheader("📈 Distribution des variables")

    df.hist(bins=15, layout=(3, 3), figsize=(15, 10))
    plt.suptitle("Distribution des variables")
    plt.tight_layout()
    st.pyplot(plt.gcf())
