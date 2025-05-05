import streamlit as st
import plotly.express as px

st.title("Heart Attack Prediction.")
#common
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error,confusion_matrix, ConfusionMatrixDisplay, f1_score, r2_score, accuracy_score, classification_report
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.decomposition import PCA

#Random Forest
from sklearn.ensemble import RandomForestClassifier

#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

#ANN 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance

# Load the dataset
csv_url = 'https://raw.githubusercontent.com/ctsfy/Data-Analytics-Heart-Attack-Prediction/refs/heads/main/heart_attack_indonesia.csv'
df = pd.read_csv(csv_url)

# Convert relevant columns to numeric
numerical_cols = ['Age', 'Cholesterol', 'BMI', 'SleepHours', 'AirQualityIndex']
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with any remaining missing values

features_na = [features for features in df.columns if df[features].isnull().sum() > 0]

if features_na:
    for feature in features_na:
        print(feature, df[feature].isnull().sum(), 'missing')
else:
    print("No missing value")

arr = []

for index, value in df['AlcoholConsumption'].items():
    if pd.isna(value):
        arr.append(index)

df = df.drop(arr)
y = df['HeartAttack']

# Convert HeartAttack to boolean
df['HeartAttack'] = df['HeartAttack'].map({'Yes': True, 'No': False})

# Setup Streamlit pages
# st.set_page_config(page_title="Heart Attack Dashboard", layout="wide")

page = st.sidebar.selectbox("Select Page", ["Filtered Data", "Univariate EDA", "Bivariate EDA", "Heatmap_All_Features", "Heatmap_Numerical_Columns", "Prediction",])

# Filtered Data Page ================================================================================================= #
if page == "Filtered Data":
    st.title("Data Overview")
    st.markdown("Use the leftbar to filter data.")
    
    heart_attack = st.sidebar.radio("Show Heart Attack Cases", ["All", "Yes", "No"])
    selected_states = st.sidebar.multiselect("Select State(s)", df['State'].unique())
    age = st.sidebar.multiselect("Select Age(s)", df['Age'].unique())
    selected_genders = st.sidebar.multiselect("Select Gender(s)", df['Gender'].unique())
    cholesterol = st.sidebar.multiselect("Select Cholesterol", df['Cholesterol'].unique())
    smoking_habits = st.sidebar.multiselect("Select Smoking Habits", df['SmokingHabits'].unique())
    physical_activity = st.sidebar.multiselect("Select Physical Activity Levels", df['PhysicalActivity'].unique())
    bmi = st.sidebar.multiselect("Select Body Mass Index", df['BMI'].unique())
    hypertension = st.sidebar.multiselect("Filter by Hypertension", df['Hypertension'].unique())
    diabetes = st.sidebar.multiselect("Filter by Diabetes", df['Diabetes'].unique())
    alcohol_consumption = st.sidebar.multiselect("Select Alcohol Consumption", df['AlcoholConsumption'].unique())
    diet_type = st.sidebar.multiselect("Select Diet Type", df['DietType'].unique())
    occupation = st.sidebar.multiselect("Select Occupation Type", df['OccupationType'].unique())
    stress_level = st.sidebar.multiselect("Select Stress Level", df['StressLevel'].unique())
    edu_level =  st.sidebar.multiselect("Select Education Level", df['EducationLevel'].unique())
    marital_status = st.sidebar.multiselect("Select Marital Status", df['MaritalStatus'].unique())
    fam_history_hd = st.sidebar.multiselect("Select Family History Heart Disease", df['FamilyHistoryHeartDisease'].unique())
    income_level = st.sidebar.multiselect("Select Income Level", df['IncomeLevel'].unique())
    healthcare_access = st.sidebar.multiselect("Select Healthcare Access", df['HealthcareAccess'].unique())
    sleep_hours = st.sidebar.multiselect("Select Sleep Hours", df['SleepHours'].unique())
    UrbanOrRural = st.sidebar.multiselect("Select Urban or Rural", df['UrbanOrRural'].unique())
    air_quality_index = st.sidebar.multiselect("Select Air Quality Index", df['AirQualityIndex'].unique())
    pollution_level = st.sidebar.multiselect("Select Pollution Level", df['PollutionLevel'].unique())
    employment_status = st.sidebar.multiselect("Filter by Employment Status", df['EmploymentStatus'].unique())

    filtered_df = df.copy()  

    if selected_states:
        filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]
    if age:
        filtered_df = filtered_df[filtered_df['Age'].isin(age)]
    if selected_genders:
        filtered_df = filtered_df[filtered_df['Gender'].isin(selected_genders)]
    if heart_attack == "Yes":
        filtered_df = filtered_df[filtered_df['HeartAttack'] == True]
    elif heart_attack == "No":
        filtered_df = filtered_df[filtered_df['HeartAttack'] == False]
    if cholesterol:
        filtered_df = filtered_df[filtered_df['Cholesterol'].isin(cholesterol)]
    if smoking_habits:
        filtered_df = filtered_df[filtered_df['SmokingHabits'].isin(smoking_habits)]
    if physical_activity:
        filtered_df = filtered_df[filtered_df['PhysicalActivity'].isin(physical_activity)]
    if bmi:
        filtered_df = filtered_df[filtered_df['BMI'].isin(bmi)]
    if hypertension:
        filtered_df = filtered_df[filtered_df['Hypertension'].isin(hypertension)]
    if diabetes:
        filtered_df = filtered_df[filtered_df['Diabetes'].isin(diabetes)]
    if alcohol_consumption:
        filtered_df = filtered_df[filtered_df['AlcoholConsumption'].isin(alcohol_consumption)]
    if diet_type:
        filtered_df = filtered_df[filtered_df['DietType'].isin(diet_type)]
    if occupation:
        filtered_df = filtered_df[filtered_df['OccupationType'].isin(occupation)]
    if stress_level:
        filtered_df = filtered_df[filtered_df['StressLevel'].isin(stress_level)]
    if edu_level:
        filtered_df = filtered_df[filtered_df['EducationLevel'].isin(edu_level)]
    if marital_status:
        filtered_df = filtered_df[filtered_df['MaritalStatus'].isin(marital_status)]
    if fam_history_hd:
        filtered_df = filtered_df[filtered_df['FamilyHistoryHeartDisease'].isin(fam_history_hd)]
    if income_level:
        filtered_df = filtered_df[filtered_df['IncomeLevel'].isin(income_level)]
    if healthcare_access:
        filtered_df = filtered_df[filtered_df['HealthcareAccess'].isin(healthcare_access)]
    if sleep_hours:
        filtered_df = filtered_df[filtered_df['SleepHours'].isin(sleep_hours)]
    if UrbanOrRural:
        filtered_df = filtered_df[filtered_df['UrbanOrRural'].isin(UrbanOrRural)]    
    if air_quality_index:
        filtered_df = filtered_df[filtered_df['AirQualityIndex'].isin(air_quality_index)]
    if pollution_level:
        filtered_df = filtered_df[filtered_df['PollutionLevel'].isin(pollution_level)]    
    if employment_status:
        filtered_df = filtered_df[filtered_df['EmploymentStatus'].isin(employment_status)]

    st.dataframe(filtered_df)

elif page == "Univariate EDA":
    st.title("Univariate Feature Distributions")

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'bool':
            fig = px.histogram(df, x=col, title=f'Distribution of {col}')
        else:
            fig = px.histogram(df, x=col, nbins=10, title=f'Distribution of {col}')
            fig.update_layout(bargap=0.1)
        st.plotly_chart(fig)

elif page == "Bivariate EDA":
    st.title("Bivariate Analysis: Feature vs HeartAttack")

    for col in df.columns:
        if col != 'HeartAttack':
            if df[col].dtype == 'object':
                fig = px.histogram(df, x=col, color='HeartAttack', barmode='group', title=f'{col} vs Heart Attack')
                fig.update_layout(bargap=0.1)
            else:
                fig = px.box(df, x='HeartAttack', y=col, title=f'{col} Distribution by Heart Attack')
            st.plotly_chart(fig)
elif page == "Heatmap_All_Features" : 
    st.title("Correlation Matrix Heatmap")

    # Make a copy of the dataset
    corr_data = df.copy()

    # Encode categorical and boolean columns safely
    for col in corr_data.columns:
        if corr_data[col].dtype == 'object' or corr_data[col].dtype == 'bool':
            try:
                le = LabelEncoder()
                corr_data[col] = le.fit_transform(corr_data[col].astype(str))
            except Exception as e:
                st.warning(f"Could not encode {col}: {e}")
                corr_data.drop(columns=[col], inplace=True)

    # Keep only numeric columns
    corr_data = corr_data.select_dtypes(include=['number'])

    # Compute correlation matrix
    corr_matrix = corr_data.corr()

    # Generate the Plotly heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title="Correlation Matrix Heatmap (Including Encoded Categorical Columns)",
        labels={'color': 'Correlation'},
        aspect="auto"
    )

    # Display the heatmap in Streamlit
    st.plotly_chart(fig_corr, use_container_width=True)

elif page == "Heatmap_Numerical_Columns" : 
    st.title("Correlation Matrix Heatmap")

    # Ensure 'HeartAttack' is numeric
    df['HeartAttack'] = df['HeartAttack'].replace({'Yes': 1, 'No': 0})

    # Select only numeric columns
    corr_data = df.select_dtypes(include=['int64', 'float64'])

    # Calculate correlation matrix
    corr_matrix = corr_data.corr()

    # Plot heatmap using seaborn
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', center=0, annot=True, ax=ax)
    ax.set_title("Correlation Matrix Heatmap")

    # Display plot in Streamlit
    st.pyplot(fig)
elif page == "Prediction":
    st.title("Heart Attack Prediction")

    #==============================================================================================================#
    st.title("Normal Random Forest")

    # --- Preprocessing ---
    X = df.drop(columns=['HeartAttack', 'ID'])
    y = df['HeartAttack']

    X = pd.get_dummies(X, drop_first=True)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, stratify=y_temp, random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Model
    rf_model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=1)
    rf_model.fit(X_train_scaled, y_train)

    # Predictions
    # y_train_pred = rf_model.predict(X_train_scaled)
    y_val_pred = rf_model.predict(X_val_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)

    # --- Evaluation: Validation ---
    st.subheader("✅ Validation Evaluation")
    st.write("Accuracy:", accuracy_score(y_val, y_val_pred))
    st.write("F1 Score:", f1_score(y_val, y_val_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_val, y_val_pred, zero_division=0))

    # --- Evaluation: Testing ---
    st.subheader("✅ Testing Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
    st.write("F1 Score:", f1_score(y_test, y_test_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_test_pred, zero_division=0))

    st.subheader("🧾 Confusion Matrix (Table View)")
    cm_table = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)

#**********************************************************************#
    st.title("Random Forest + SMOTE oversampling")

    converted_HeartAttack = {'HeartAttack': {'Yes': 1, "No": 0}}
    df = df.replace(converted_HeartAttack)

    # Split the dataset
    X = df.drop(columns=['HeartAttack'])
    y = df['HeartAttack']

    # One-hot encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-test-validation split
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42) # 40% training 60% temp
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, random_state=42) #50% from temp used for testing

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Build and train Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=1, min_samples_split=100, max_features=2)
    rf_model.fit(X_train_scaled, y_train_smote)

    # Make predictions
    y_train_pred = rf_model.predict(X_train_scaled)
    y_val_pred = rf_model.predict(X_val_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)

    # Display evaluation metrics
    # st.subheader("==== Training Data Evaluation ====")
    # st.write("Accuracy:", accuracy_score(y_train_smote, y_train_pred))
    # st.write("F1 Score:", f1_score(y_train_smote, y_train_pred, zero_division=0))
    # st.write("Classification Report:\n", classification_report(y_train_smote, y_train_pred, zero_division=0))

    st.subheader("\n==== Validation Data Evaluation ====")
    st.write("Accuracy:", accuracy_score(y_val, y_val_pred))
    st.write("F1 Score:", f1_score(y_val, y_val_pred, zero_division=0))
    st.write("Classification Report:\n", classification_report(y_val, y_val_pred, zero_division=0))

    st.subheader("\n==== Testing Data Evaluation ====")
    st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
    st.write("F1 Score:", f1_score(y_test, y_test_pred, zero_division=0))
    st.write("Classification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))
    
    st.subheader("🧾 Confusion Matrix (Table View)")
    cm_table = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)

#*****************************************************************************#

    st.title("Random Forest + undersampling")

    converted_HeartAttack = {'HeartAttack': {'Yes': 1, "No": 0}}
    df = df.replace(converted_HeartAttack)

    # Split features and target
    X = df.drop(columns=['HeartAttack'])
    y = df['HeartAttack']

    # One-hot encoding for categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Split into train, validation, test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, stratify=y_temp, random_state=42)

    # Undersampling
    train_data = pd.concat([X_train, y_train], axis=1)
    majority = train_data[train_data['HeartAttack'] == 0]
    minority = train_data[train_data['HeartAttack'] == 1]
    majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
    train_downsampled = pd.concat([majority_downsampled, minority]).sample(frac=1, random_state=42).reset_index(drop=True)

    X_train_down = train_downsampled.drop(columns=['HeartAttack'])
    y_train_down = train_downsampled['HeartAttack']

    st.subheader("Class Distribution After Undersampling")
    st.write(y_train_down.value_counts())

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_down)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train_down)

    # Predictions
    y_train_pred = rf_model.predict(X_train_scaled)
    y_val_pred = rf_model.predict(X_val_scaled)
    y_test_pred = rf_model.predict(X_test_scaled)

    # Evaluation Outputs
    # st.subheader("Training Data Evaluation")
    # st.write("Accuracy:", accuracy_score(y_train_down, y_train_pred))
    # st.write("F1 Score:", f1_score(y_train_down, y_train_pred))
    # st.text("Classification Report:\n" + classification_report(y_train_down, y_train_pred, zero_division=0))

    st.subheader("Validation Data Evaluation")
    st.write("Accuracy:", accuracy_score(y_val, y_val_pred))
    st.write("F1 Score:", f1_score(y_val, y_val_pred))
    st.text("Classification Report:\n" + classification_report(y_val, y_val_pred, zero_division=0))

    st.subheader("Testing Data Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
    st.write("F1 Score:", f1_score(y_test, y_test_pred))
    st.text("Classification Report:\n" + classification_report(y_test, y_test_pred, zero_division=0))

    st.subheader("🧾 Confusion Matrix (Table View)")
    cm_table = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)

#********************************************************************************#

    st.title("Random Forest + oversampling + PCA + hyperparameter tuning")

    converted_HeartAttack = {
        'HeartAttack': {'Yes': 1, "No": 0}
    }

    df = df.replace(converted_HeartAttack)

    X = df.drop(columns=['HeartAttack', 'ID'])
    y = df['HeartAttack']

    X = pd.get_dummies(X, drop_first=True)

    #split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, random_state=42)

    #smote
    smote = SMOTE(random_state = 42) 
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    #scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)

    #pca
    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X_train_scaled)

    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
    df_pca['HeartAttack'] = y_train_smote.reset_index(drop=True)
    print(df_pca.head())

    X_train_final = df_pca.drop(columns=['HeartAttack'])
    y_train_final = df_pca['HeartAttack']

    #testing and validation
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    rf_tuned = RandomForestClassifier(n_estimators= 150, min_samples_split= 10, min_samples_leaf= 4, max_features= 'sqrt', max_depth= 20, random_state=42)
    rf_tuned.fit(X_train_final, y_train_final)

    y_test_rf_pred = rf_tuned.predict(X_test_pca)
    y_val_rf_pred = rf_tuned.predict(X_val_pca)

    accuracy_test_rf = accuracy_score(y_test, y_test_rf_pred)
    accuracy_val_rf = accuracy_score(y_val, y_val_rf_pred)

    classification_report_test_rf = classification_report(y_test, y_test_rf_pred, zero_division=0)
    classification_report_val_rf = classification_report(y_val, y_val_rf_pred, zero_division=0)

    st.write("Accuracy (Validation):", accuracy_val_rf)
    st.write("Accuracy (Test):", accuracy_test_rf)
    st.write("\nClassification Report (Validation):\n", classification_report_val_rf)
    st.write("\nClassification Report (Test):\n", classification_report_test_rf)

    cm = confusion_matrix(y_test, y_test_rf_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots()
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    plt.title("Confusion Matrix - Random Forest")

    st.pyplot(fig)

    cm_table = confusion_matrix(y_test, y_test_rf_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)

#********************************************************************************#


    st.title("Normal Logistic Regression")

    converted_HeartAttack = {
        'HeartAttack': {'Yes': 1, 'No': 0}
    }
    df = df.replace(converted_HeartAttack)

    X = df.drop(columns=['HeartAttack'])
    y = df['HeartAttack']
    X = pd.get_dummies(X, drop_first=True)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Splitting
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, random_state=42)

    # Re-scaling just in case
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Model Training
    logReg = LogisticRegression(max_iter=1000, random_state=42)
    logReg.fit(X_train_scaled, y_train)

    # Predictions
    y_train_pred = logReg.predict(X_train_scaled)
    y_val_pred = logReg.predict(X_val_scaled)
    y_test_pred = logReg.predict(X_test_scaled)

    # Probability Percentage
    HA_Yes_percentage = np.mean(y_test_pred) * 100
    HA_No_percentage = 100 - HA_Yes_percentage

    # st.subheader("📈 Heart Attack Probability Prediction")
    # st.write(f"Probability of HeartAttack = Yes: {HA_Yes_percentage:.2f}%")
    # st.write(f"Probability of HeartAttack = No: {HA_No_percentage:.2f}%")

    # Evaluation - Train
    # st.subheader("✅ Training Data Evaluation")
    # st.write("Accuracy:", accuracy_score(y_train, y_train_pred))
    # st.write("F1 Score:", f1_score(y_train, y_train_pred, zero_division=0))
    # st.text("Classification Report:")
    # st.text(classification_report(y_train, y_train_pred, zero_division=0))

    # Evaluation - Validation
    st.subheader("✅ Validation Data Evaluation")
    st.write("Accuracy:", accuracy_score(y_val, y_val_pred))
    st.write("F1 Score:", f1_score(y_val, y_val_pred, zero_division=0))
    st.text("Classification Report:")
    st.text(classification_report(y_val, y_val_pred, zero_division=0))

    # Evaluation - Test
    st.subheader("✅ Testing Data Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
    st.write("F1 Score:", f1_score(y_test, y_test_pred, zero_division=0))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_test_pred, zero_division=0))

    st.subheader("🧾 Confusion Matrix (Table View)")
    cm_table = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)

#*****************************************************************************#

    st.title("Logistic Regression + oversampling")

    converted_HeartAttack = {
        'HeartAttack': {'Yes': 1, "No": 0}
    }
    df = df.replace(converted_HeartAttack)

    # Feature-target split
    X = df.drop(columns=['HeartAttack'])
    y = df['HeartAttack']

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Train/val/test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, random_state=42)
  
    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
  
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Model training
    logReg = LogisticRegression(max_iter=1000, random_state=42)
    logReg.fit(X_train_scaled, y_train_smote)

    # Predictions
    y_train_pred = logReg.predict(X_train_scaled)
    y_val_pred = logReg.predict(X_val_scaled)
    y_test_pred = logReg.predict(X_test_scaled)

    # Probability prediction
    y_test_proba = logReg.predict_proba(X_test_scaled)[:, 1]
    HA_Yes_percentage = np.mean(y_test_pred) * 100
    HA_No_percentage = 100 - HA_Yes_percentage

    # st.subheader("Heart Attack Probability Prediction")
    # st.write(f"Probability of HeartAttack = Yes: {HA_Yes_percentage:.2f}%")
    # st.write(f"Probability of HeartAttack = No: {HA_No_percentage:.2f}%")

    # Evaluation metrics
    # st.subheader("Training Evaluation")
    # st.write("Accuracy:", accuracy_score(y_train_smote, y_train_pred))
    # st.write("F1 Score:", f1_score(y_train_smote, y_train_pred, zero_division=0))
    # st.text(classification_report(y_train_smote, y_train_pred, zero_division=0))

    st.subheader("Validation Evaluation")
    st.write("Accuracy:", accuracy_score(y_val, y_val_pred))
    st.write("F1 Score:", f1_score(y_val, y_val_pred, zero_division=0))
    st.text(classification_report(y_val, y_val_pred, zero_division=0))

    st.subheader("Testing Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
    st.write("F1 Score:", f1_score(y_test, y_test_pred, zero_division=0))
    st.text(classification_report(y_test, y_test_pred, zero_division=0))

    st.subheader("🧾 Confusion Matrix (Table View)")
    cm_table = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)

#******************************************************************************#

    st.title("Logistic Regression + undersampling")

    converted_HeartAttack = {'HeartAttack': {'Yes': 1, "No": 0}}
    df = df.replace(converted_HeartAttack)

    X = df.drop(columns=['HeartAttack'])
    y = df['HeartAttack']

    X = pd.get_dummies(X, drop_first=True)

    # --- Split Data ---
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, stratify=y_temp, random_state=42)

    # --- Undersample ---
    train_data = pd.concat([X_train, y_train], axis=1)
    majority = train_data[train_data['HeartAttack'] == 0]
    minority = train_data[train_data['HeartAttack'] == 1]

    majority_downsampled = resample(majority,
                                    replace=False,
                                    n_samples=len(minority),
                                    random_state=42)

    train_downsampled = pd.concat([majority_downsampled, minority])
    train_downsampled = train_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train_down = train_downsampled.drop(columns=['HeartAttack'])
    y_train_down = train_downsampled['HeartAttack']

    st.subheader("[Undersampling] Class Distribution")
    st.write(y_train_down.value_counts())

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_down)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Train Logistic Regression ---
    logreg_under = LogisticRegression(max_iter=1000, random_state=42)
    logreg_under.fit(X_train_scaled, y_train_down)

    # --- Predict ---
    y_train_pred = logreg_under.predict(X_train_scaled)
    y_val_pred = logreg_under.predict(X_val_scaled)
    y_test_pred = logreg_under.predict(X_test_scaled)

    # --- Evaluation ---
    # st.subheader("Training Evaluation")
    # st.write("Accuracy:", accuracy_score(y_train_down, y_train_pred))
    # st.text(classification_report(y_train_down, y_train_pred, zero_division=0))

    st.subheader("Validation Evaluation")
    st.write("Accuracy:", accuracy_score(y_val, y_val_pred))
    st.text(classification_report(y_val, y_val_pred, zero_division=0))

    st.subheader("Testing Evaluation")
    st.write("Accuracy:", accuracy_score(y_test, y_test_pred))
    st.text(classification_report(y_test, y_test_pred, zero_division=0))

    st.subheader("🧾 Confusion Matrix (Table View)")
    cm_table = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)

#**********************************************************************#

    st.title("Logistic Regression + oversampling + PCA + hyperparameter tuning")

    converted_HeartAttack = {
        'HeartAttack': {'Yes': 1, "No": 0}
    }

    df = df.replace(converted_HeartAttack)

    X = df.drop(columns=['HeartAttack', 'ID'])
    y = df['HeartAttack']

    X = pd.get_dummies(X, drop_first=True)

    #split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, random_state=42)

    #smote
    smote = SMOTE(random_state = 42) #oversampling method: SMOTE
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    #scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)

    #pca
    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X_train_scaled)

    df_pca = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
    df_pca['HeartAttack'] = y_train_smote.reset_index(drop=True)
    print(df_pca.head())

    X_train_final = df_pca.drop(columns=['HeartAttack'])
    y_train_final = df_pca['HeartAttack']

    #testing and validation
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    logReg_tuned = LogisticRegression(solver='saga', penalty='l1', max_iter=1000, C=np.float64(0.004641588833612777), random_state=42)
    logReg_tuned.fit(X_train_final, y_train_final)

    y_test_logReg_pred = logReg_tuned.predict(X_test_pca)
    y_val_logReg_pred = logReg_tuned.predict(X_val_pca)

    accuracy_test_logReg = accuracy_score(y_test, y_test_logReg_pred)
    accuracy_val_logReg = accuracy_score(y_val, y_val_logReg_pred)

    print("LOGISTIC REGRESSION")
    print("---------------------")
    classification_report_test_logReg = classification_report(y_test, y_test_logReg_pred, zero_division=0)
    classification_report_val_logReg = classification_report(y_val, y_val_logReg_pred, zero_division=0)

    print("Accuracy (Validation):", accuracy_val_logReg)
    print("Accuracy (Test):", accuracy_test_logReg)
    print("\nClassification Report (Validation):\n", classification_report_val_logReg)
    print("\nClassification Report (Test):\n", classification_report_test_logReg)

    # cm = confusion_matrix(y_test, y_test_logReg_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # fig, ax = plt.subplots()
    # disp.plot(ax=ax, cmap='Blues', colorbar=False)
    # plt.title("Confusion Matrix - Random Forest")
    
    # st.pyplot(fig)

    st.subheader("🧾 Confusion Matrix (Table View)")
    cm_table = confusion_matrix(y_test, y_test_logReg_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)

#**********************************************************************#
    st.title("Normal Artificial Neural Network")

    converted_HeartAttack = {'HeartAttack': {'Yes': 1, "No": 0}}
    df = df.replace(converted_HeartAttack)

    X = df.drop(columns=['HeartAttack'])
    y = df['HeartAttack']
    X = pd.get_dummies(X, drop_first=True)

    # --- Split Data ---
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, stratify=y_temp, random_state=42)

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Build ANN Model ---
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        # Dropout(0.3),
        Dense(32, activation='relu'),
        # Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # --- Train Model ---
    history = model.fit(X_train_scaled, y_train,
                        validation_data=(X_val_scaled, y_val),
                        epochs=5, batch_size=32, verbose=0)

    # --- Predict ---
    y_train_pred = (model.predict(X_train_scaled) > 0.5).astype(int)
    y_val_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
    y_test_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    # --- Evaluation ---
    # st.subheader("Testing Accuracy")
    # st.write("Accuracy:", accuracy_score(y_test, y_test_pred))

    # st.subheader("Training Evaluation")
    # st.text(classification_report(y_train, y_train_pred, zero_division=0))

    st.subheader("Validation Evaluation")
    st.text(classification_report(y_val, y_val_pred, zero_division=0))

    st.subheader("Testing Evaluation")
    st.text(classification_report(y_test, y_test_pred, zero_division=0))

    st.subheader("🧾 Confusion Matrix (Table View)")
    cm_table = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)

#***********************************************************************#

    st.title("Artificial Neural Network + oversampling")

    converted_HeartAttack = {'HeartAttack': {'Yes': 1, "No": 0}}
    df = df.replace(converted_HeartAttack)

    X = df.drop(columns=['HeartAttack'])
    y = df['HeartAttack']
    X = pd.get_dummies(X, drop_first=True)

    # --- Split Data ---
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, stratify=y_temp, random_state=42)

    # --- Apply SMOTE ---
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    st.subheader("[SMOTE] Class distribution after oversampling:")
    st.write(y_train_smote.value_counts())

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Build ANN Model ---
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # --- Train Model ---
    history = model.fit(X_train_scaled, y_train_smote,
                        validation_data=(X_val_scaled, y_val),
                        epochs=5, batch_size=32, verbose=0)

    # --- Predict ---
    y_train_pred = (model.predict(X_train_scaled) > 0.5).astype(int)
    y_val_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
    y_test_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    # --- Evaluation ---
    st.subheader("==== ANN (SMOTE Oversampling) ====")

    # st.subheader("Training Evaluation")
    # st.text(classification_report(y_train_smote, y_train_pred, zero_division=0))

    st.subheader("Validation Evaluation")
    st.text(classification_report(y_val, y_val_pred, zero_division=0))

    st.subheader("Testing Evaluation")
    st.text(classification_report(y_test, y_test_pred, zero_division=0))

    # --- Accuracy Score ---
    st.subheader("Final Test Accuracy")
    st.write("Accuracy:", accuracy_score(y_test, y_test_pred))

    st.subheader("🧾 Confusion Matrix (Table View)")
    cm_table = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)

#**********************************************************************#

    st.title("Artificial Neural Network + undersampling")

    converted_HeartAttack = {'HeartAttack': {'Yes': 1, "No": 0}}
    df = df.replace(converted_HeartAttack)

    X = df.drop(columns=['HeartAttack'])
    y = df['HeartAttack']

    X = pd.get_dummies(X, drop_first=True)

    # --- Split Data ---
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.6666, stratify=y_temp, random_state=42)

    # --- Undersample ---
    train_data = pd.concat([X_train, y_train], axis=1)
    majority = train_data[train_data['HeartAttack'] == 0]
    minority = train_data[train_data['HeartAttack'] == 1]

    majority_downsampled = resample(majority,
                                    replace=False,
                                    n_samples=len(minority),
                                    random_state=42)

    train_downsampled = pd.concat([majority_downsampled, minority])
    train_downsampled = train_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

    X_train_down = train_downsampled.drop(columns=['HeartAttack'])
    y_train_down = train_downsampled['HeartAttack']

    st.subheader("[Undersampling] Class distribution after undersampling:")
    st.write(y_train_down.value_counts())

    # --- Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_down)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # --- Build ANN Model ---
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # --- Train Model ---
    history = model.fit(X_train_scaled, y_train_down,
                        validation_data=(X_val_scaled, y_val),
                        epochs=50, batch_size=32, verbose=0)

    # --- Predict ---
    y_train_pred = (model.predict(X_train_scaled) > 0.5).astype(int)
    y_val_pred = (model.predict(X_val_scaled) > 0.5).astype(int)
    y_test_pred = (model.predict(X_test_scaled) > 0.5).astype(int)

    # --- Evaluation ---
    st.subheader("==== ANN (Undersampling) ====")

    # st.subheader("Training Evaluation")
    # st.text(classification_report(y_train_down, y_train_pred, zero_division=0))

    st.subheader("Validation Evaluation")
    st.text(classification_report(y_val, y_val_pred, zero_division=0))

    st.subheader("Testing Evaluation")
    st.text(classification_report(y_test, y_test_pred, zero_division=0))

    # --- Accuracy Score ---
    st.subheader("Final Test Accuracy")
    st.write("Accuracy:", accuracy_score(y_test, y_test_pred))

    st.subheader("🧾 Confusion Matrix (Table View)")
    cm_table = confusion_matrix(y_test, y_test_pred)
    cm_df = pd.DataFrame(cm_table, 
                         columns=['Predicted No', 'Predicted Yes'], 
                         index=['Actual No', 'Actual Yes'])
    st.write(cm_df)


