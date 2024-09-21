import streamlit as st
import datetime
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle as pk
import os

# ML-related imports
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle
import warnings
from scipy import stats
from scipy.stats import norm, skew

warnings.filterwarnings('ignore')

# Suppress Streamlit deprecation warnings for global use of pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)

# Define paths
train_path = "data/train.csv"
test_path = "data/test.csv"
real = "data/RealEstate.csv"
head = 'images/head.jpg'

# Load data functions with caching
@st.cache_data
def load_train_data(train_path):
    return pd.read_csv(train_path)

@st.cache_data
def load_test_data(test_path):
    return pd.read_csv(test_path)

@st.cache_data
def load_data(real):
    return pd.read_csv(real)

# Function to save data
def save_data(value, res):
    file = 'db.csv'
    if not os.path.exists(file):
        with open(file, 'w') as f:
            f.write("OverallQual,GrLivArea,GarageCars,GarageArea,TotalBsmtSF,fstFlrSF,FullBath,TotRmsAbvGrd,YearBuilt,YearRemodAdd,GarageYrBlt,MasVnrArea,Fireplaces,BsmtFinSF1,Result\n")
    with open(file, 'a') as f:
        data = f"{','.join(map(str, value))},{res}\n"
        f.write(data)

# Sidebar
st.sidebar.image(head, caption="Project on Artificial Intelligence", use_column_width=True)

# Main title
st.title("House Pricing Analysis")

# Menu options
menu = ["House Prediction", "Predicted House", "About", "Visual"]
choices = st.sidebar.selectbox("Menu Bar", menu)

if choices == 'House Prediction':
    st.subheader("House Prediction")
    OverallQual = st.selectbox("Select the overall quality (10 being 'Very Excellent' and 1 being 'Very Poor')", range(10, 0, -1))
    GrLivArea = st.number_input("Enter Ground Floor Living Area (in Sqft)", value=0, min_value=0, format='%d')
    GarageArea = st.number_input("Enter area of Garage (in Sqft)", value=0.0, format='%f', step=1.0)
    GarageCars = st.number_input("Number of Cars to be accommodated in garage", min_value=1.0, max_value=10.0, step=1.0, format='%f')
    TotalBsmtSF = st.number_input("Enter area of Basement (in Sqft)", value=0.0, format='%f', step=1.0)
    fstFlrSF = st.number_input("Enter area of First Floor (in Sqft)", value=0, format='%d')
    FullBath = st.number_input("Enter number of Bathrooms", min_value=1, max_value=10, format='%d')
    TotRmsAbvGrd = st.number_input("Enter number of Rooms", min_value=1, max_value=10, format='%d')
    years = tuple(range(1872, 2011))
    YearBuilt = st.selectbox("Select the year the house was built", years)
    remyears = tuple(range(1950, 2011))
    YearRemodAdd = st.selectbox("Select Remodel date (same as construction date if no remodeling or additions)", remyears)
    garyears = tuple(map(float, range(1872, 2011)))
    GarageYrBlt = st.selectbox("Select year in which Garage was built", garyears)
    MasVnrArea = st.number_input("Masonry veneer area (in Sqft)", value=0.0, format='%f', step=1.0)
    Fireplaces = st.number_input("Select number of Fireplaces", min_value=1, max_value=10, format='%d')
    BsmtFinSF1 = st.number_input("Enter Basement Finished Area (in Sqft)", value=0, format='%d')
    submit = st.button('Predict')

    if submit:
        st.success("Prediction Done")
        value = [OverallQual, GrLivArea, GarageCars, GarageArea, TotalBsmtSF, fstFlrSF, FullBath, TotRmsAbvGrd, YearBuilt, YearRemodAdd, GarageYrBlt, MasVnrArea, Fireplaces, BsmtFinSF1]
        df = pd.DataFrame(value).transpose()
        model = pk.load(open('model & scaler/rfrmodel.pkl', 'rb'))
        scaler = pk.load(open('model & scaler/scale.pkl', 'rb'))
        df = scaler.transform(df)
        ans = int(model.predict(df)) * 5
        st.subheader(f"The price is {ans} (INR)")
        save_data(value, ans)

if choices == 'Predicted House':
    st.subheader("Predicted House")
    st.info("Expand to see data clearly")
    if os.path.exists("db.csv"):
        data = pd.read_csv('db.csv')
        st.write(data)
    else:
        st.error("Please try some prediction, then the data will be available here")

if choices == 'About':
    st.subheader("About Us")
    info = '''
        A house value is simply more than location and square footage. Like the features that make up a person, an educated party would want to know all aspects that give a house its value.

        We are going to take advantage of all of the feature variables available to use and use it to analyze and predict house prices.

        We are going to break everything into logical steps that allow us to ensure the cleanest, most realistic data for our model to make accurate predictions from.

        - Load Data and Packages
        - Analyzing the Test Variable (Sale Price)
        - Multivariable Analysis
        - Impute Missing Data and Clean Data
        - Feature Transformation/Engineering
        - Modeling and Predictions
    '''
    st.markdown(info, unsafe_allow_html=True)

if choices == 'Visual':
    st.subheader("Data Visualization")

    train_data = load_train_data(train_path)
    test_data = load_test_data(test_path)

    if st.checkbox("View dataset column description"):
        st.subheader('Displaying the column-wise stats for the dataset')
        st.write(train_data.columns)
        st.write(train_data.describe())

    st.subheader('Correlation between dataset columns')
    corrmatrix = train_data.corr()
    f, ax = plt.subplots(figsize=(20, 9))
    sns.heatmap(corrmatrix, vmax=0.8, annot=True)
    st.pyplot(f)

    st.subheader("Most correlated features")
    top_corr_feat = corrmatrix.index[abs(corrmatrix['SalePrice']) > 0.5]
    plt.figure(figsize=(10, 10))
    sns.heatmap(train_data[top_corr_feat].corr(), annot=True, cmap="RdYlGn")
    st.pyplot()

    st.subheader("Comparing Overall Quality vs Sale Price")
    f, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='OverallQual', y='SalePrice', data=train_data, ax=ax)
    st.pyplot(f)

    st.subheader("Pairplot visualization to describe correlation easily")
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(train_data[cols], height=2.5)
    st.pyplot()

    st.subheader("Analysis of Sale Price column in dataset")
    f, ax = plt.subplots(figsize=(10, 6))
    sns.distplot(train_data['SalePrice'], fit=norm, ax=ax)
    (mu, sigma) = norm.fit(train_data['SalePrice'])
    st.write('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')
    st.pyplot(f)

    fig = plt.figure(figsize=(10, 10))
    res = stats.probplot(train_data['SalePrice'], plot=plt)
    st.pyplot(fig)
