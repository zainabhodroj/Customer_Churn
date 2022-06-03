import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hydralit_components as hc
import plotly.graph_objects as go
import time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
#import imblearn
#from imblearn.combine import SMOTETomek
#from imblearn.under_sampling import TomekLinks
import requests
from streamlit_lottie import st_lottie
import json

#defining lottie function to visualize animated pictures
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def upload():

    # Dispaly Upload File Widget
    uploaded = st.file_uploader(label="Upload your Own Data!", type=["csv"])

    # Save the file in internal memory of streamlit
    if 'file' not in st.session_state:
        st.session_state['file'] = None


    st.session_state['file'] = uploaded

    if 'table' not in st.session_state:
        st.session_state['table'] = None 
        
    if uploaded is not None:
        st.session_state['table'] = pd.read_csv(uploaded)
        return st.session_state['table']
    else:
        st.session_state['table'] = pd.read_csv('CleanCustomerChurn.csv')
        return st.session_state['table']



#setting configuration of the page and expanding it
st.set_page_config(layout='wide', initial_sidebar_state='collapsed', page_title='Customer Churn Prediction')
st.expander('Expander')


#creating menu data which will be used in navigation bar specifying the pages of the dashboard
menu_data = [
    {'label': "Home", 'icon': 'bi bi-house'},
    {'label': 'Data', 'icon': 'bi bi-table'},
    {'label':"Exploratory Data Analysis", 'icon':'bi bi-bar-chart-line'},
    {'label':"Consumer Behavior", 'icon': 'bi bi-person-workspace'},
    {'label':"Churn Analysis", 'icon': 'bi bi-person-x'},
    {'label': 'Profiling', 'icon': 'bi bi-person-lines-fill'},
    {'label':"Log Out", 'icon': 'bi bi-person-file-earmark-fill'},
    ]

over_theme ={'txc_inactive': 'white','menu_background':'rgb(255,255,255)', 'option_active':'white'}

#inserting hydralit component: nagivation bar
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=False,
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)


