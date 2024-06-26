# Import necessary libraries
import streamlit as st
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import necessary functions from PyCaret
from pycaret.classification import setup as setup_clf,compare_models as compare_models_clf,ClassificationExperiment
from pycaret.regression import setup as setup_reg,compare_models as compare_models_reg,RegressionExperiment

# Function to display DataFrame information
def display_df_info(df):
    st.header("The Dataset Information")
    st.write('Head of Dataset:', df.head())
    st.write('Data Shape:', df.shape)
    st.write('Column Names:', df.columns.tolist())
    st.write('Column Data Types:', df.dtypes)
    st.write('Summary Statistics:', df.describe().transpose())

# Function to drop selected columns
def drop_columns(df,dropped_columns):
    st.header("Columns removal")
    st.write('dataset after column removed:', df.drop(dropped_columns, axis=1))
    df = df.drop(dropped_columns, axis=1)
    return df


# Function to handle missing values
def handle_missing_values(df,cat_feature):
    st.header("Handle Missing Values")
    le = LabelEncoder()

    # ask user to how to handle missing values for numerical columns ( mean or median or mode)
    st.header("Numerical Columns")
    num_feature = df.select_dtypes(['int64', 'float64']).columns
    HMV_num = st.radio("How do you want to handle missing values for numerical columns?", ["mean", "median","mode"])
    for col in num_feature:
        df[col] = SimpleImputer(strategy=HMV_num, missing_values=np.nan).fit_transform(df[col].values.reshape(-1, 1))
    if (len(num_feature) != 0):
        st.write(num_feature)
    
    # ask user to how to handle missing values for categorical columns ( mode or additional class)
    st.header("Categorical Columns")
    HMV_cat = st.radio("How do you want to handle missing values for categorical columns?", ['most frequent', "mode"])
    for col in cat_feature:
        if df[col].nunique() > 7:
            df[col] = SimpleImputer(strategy='HMV_cat', missing_values=np.nan).fit_transform(df[col].values.reshape(-1, 1))
        else:
            df[col] = le.fit_transform(df[col])
    if (len(cat_feature) != 0):
        st.write(cat_feature)

    if (len(cat_feature) != 0 or len(num_feature) != 0):
        st.header("Number of null values")
        st.write(df.isna().sum())
    
    return df

# Function to perform EDA
def perform_eda(df,WantedColumns):
    st.header("EDA")
    # Histograms
    st.subheader("Histograms")
    for col in WantedColumns:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram for {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        st.pyplot()
    
    # Correlation Matrix"
    st.subheader("Correlation Matrix")
    corr = df[WantedColumns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")
    st.pyplot()
    return df

# Function to encode categorical data
def encode_categorical(df,cat_feature):
    le = LabelEncoder()
    encoding_method = st.radio("How to encode categorical data?", ("Label Encoding", "One-Hot Encoding"))
    if encoding_method == "Label Encoding":
        # Apply Label Encoding to categorical columns
        for col in cat_feature:
            df[col] = le.fit_transform(df[col])
        st.write(df.head())
    else:
        encoder = OneHotEncoder(handle_unknown='ignore')
        for col in cat_feature:
            encoder_df = df(encoder.fit_transform(df[col]).toarray())
            final_df = df.join(encoder_df)
            final_df.drop(col, axis=1, inplace=True)
        st.write(df.head())

    return df

# Function to setup and compare models
def setup_and_compare_models(df, target_col):
    if df[target_col].dtype in ['int64', 'float64']:
        TheModel = "Regression"
    else:
        TheModel = "Classification"
        
    st.header("Detected Task Type", TheModel)

    # Initialize RegressionExperiment or ClassificationExperiment based on detected task type
    if TheModel == 'Regression':
        ModelExp = RegressionExperiment()
    elif TheModel == 'Classification':
        ModelExp = ClassificationExperiment()
        
    # Setup experiment with data and target variable
    ModelExp.setup(df, target=target_col, session_id=123)
    # Compare models and select best performing model
    best = ModelExp.compare_models()
    st.header("Best Algorithm")
    st.write(best)
    # Evaluate best model
    st.write(ModelExp.evaluate_model(best))
    # Make predictions using best model
    st.header("30 rows of Prediction")
    predictions = ModelExp.predict_model(best, data=df, raw_score=True)
    st.write(predictions.head(30))
    
    if pd.api.types.is_numeric_dtype(df[target_col]):
        exp_reg = setup_reg(data = df, target = target_col, session_id=123, verbose=False)
        best_model = compare_models_reg()
    else:
        exp_clf = setup_clf(data = df, target = target_col, session_id=123, verbose=False)
        best_model = compare_models_clf()
    st.write(best_model)

# Title of the app
st.title('PyCaret & Streamlit App Capstone')

# Add a sidebar
st.sidebar.title("Settings Sidebar")

# Add a file uploader to the sidebar
uploaded_file = st.sidebar.file_uploader(label="Upload your input CSV file", type=['csv'])

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)

    # Show the uploaded file
    st.write('The full Dataset:',df)

    # Display DataFrame information
    display_df_info(df)

    # ask user to drop columns if he want 
    dropped_columns = st.sidebar.multiselect("Select the columns you like to drop: ", df.columns)
    if dropped_columns:
        df = drop_columns(df,dropped_columns)
    
    # Ask the user if he would like to Handle missing values
    cat_feature = df.select_dtypes(['object']).columns
    isHMV = st.sidebar.checkbox("Handle Missing Values?")
    if isHMV:
        df = handle_missing_values(df,cat_feature)

    # Ask the user if he would like to Handle missing values
    isEDA = st.sidebar.checkbox("Perform EDA?")
    if isEDA:
        WantedColumns = st.sidebar.multiselect("What columns do you want to analyze?", options=df.columns)
        if WantedColumns:
            df = perform_eda(df,WantedColumns)
            
    # ask user to how to encode categorical data ( one hot or label encoding )        
    isECD = st.sidebar.checkbox("Encode Categorical Data?")
    if isECD:
        encode_categorical(df,cat_feature)
        
    # Ask user for the target column
    if df is not None:
        target_col = st.sidebar.selectbox("Select the target column", df.columns)
        # Setup and compare models
        isCM = st.sidebar.checkbox("Begin Comparing models?")
        if isCM:
            setup_and_compare_models(df, target_col)
