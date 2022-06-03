import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import hydralit_components as hc
import plotly.graph_objects as go
import time
import pickle
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
import streamlit as st


# Streamlit Style Settings
def webapp_style():
    hide_streamlit_style = """
                <style>
                    #MainMenu {
                                visibility: none;
                            }
                    footer {
                            visibility: hidden;
                            }
                    footer:after {
                                content:'Made by Zainab Hodroj ❤️'; 
                                visibility: visible;
                                display: block;
                                position: relative;
                                text-align: center;
                                padding: 15px;
                                top: 2px;
                                }
    
                </style>
                """
    markdown = st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    return markdown

#defining lottie function to visualize animated pictures
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def upload():

    # Dispaly Upload File Widget
    uploaded = st.file_uploader(label="Upload your own data", type=["csv"])

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
        st.session_state['table'] = pd.read_csv('Customer_Churn.csv')
        return st.session_state['table']



#setting configuration of the page and expanding it
st.set_page_config(layout='wide', initial_sidebar_state='collapsed', page_title='Customer Churn Prediction')
st.expander('Expander')


#creating menu data which will be used in navigation bar specifying the pages of the dashboard
menu_data = [
    {'label': "Home", 'icon': 'bi bi-house-fill'},
    {'label': 'Data', 'icon': 'bi bi-bar-chart'},
    {'label':"EDA", 'icon':'bi bi-search'},
    {'label':"Churn Prediction", 'icon': 'bi bi-person-fill'},
    {'label': 'Profiling', 'icon': 'bi bi-person-badge'},
    ]

over_theme = {'txc_inactive': 'white','menu_background':'rgb(112, 230, 220)', 'option_active':'white'}


#inserting hydralit component: nagivation bar
menu_id = hc.nav_bar(
    menu_definition=menu_data,
    override_theme=over_theme,
    hide_streamlit_markers=False,
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)

#editing first page of the dashboard with images, titles, and text
if menu_id == 'Home':
    col1, col2 = st.columns(2)
    #col1.image('churn.png')
    with col1:
        st.title('Customer Churn Prediction')
        st.write('Customer churn is the percentage of customers that stopped using your companys product or service during a certain time frame. It is a measure of the customer loyalty and retention.')
        st.write("Once businesses determine their customer churn rate, they can determine why customers are leaving and identify customer retention strategies that could help.")
        st.write("You will be able to have hands on experience with a model that can accurately predict if a customer is likely to churn.")
        m = st.markdown("""
        <style>
            div.stButton > button:first-child {
            color: #fff;
            background-color: rgb(112, 230, 220);
            }
        </style>""", unsafe_allow_html=True)
        st.write("---")

    with col2:
        lottie_home= load_lottiefile("home.json")
        st_lottie(lottie_home)
        



#BREAK
#editing second page which is about the data
if menu_id == 'Data':
    #add option to choose own data or given data defined earlier
    upload()
    col1, col2 = st.columns([2,1])

    with col1:
        lottie_data= load_lottiefile("exploratory.json")
        st_lottie(lottie_data)

    with col2:
        st.title("Let's Take a Look at the Data")
        st.markdown("""
        <style>
        .change-font {
        font-size:20px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<p class="change-font">Each customer has churn value that estimates if they will leave or not. This value is based on features like User demographic information, Browsing behavior, and Historical purchase data among other information. </p>', unsafe_allow_html=True)
        st.markdown('<p class="change-font">The values are between 1 and 5. </p>', unsafe_allow_html=True)
        data= pd.read_csv(r'Customer_Churn.csv')
        st.write("Data was obtained from the Kaggle-[Churn Risk Rate](https://www.kaggle.com/datasets/imsparsh/churn-risk-rate-hackerearth-ml) ", unsafe_allow_html=True)
        if st.checkbox('Dataset'):
            st.dataframe(data.head(5))
        if st.checkbox('Statistics'):
            st.dataframe(data.describe())
    


df= pd.read_csv(r'Customer_Churn.csv')
df['gender'].replace("F", 'Female',inplace=True)
df['gender'].replace("M", 'Male',inplace=True)

# 3rd page Exploratory Data Analysis
if menu_id == 'EDA':
    col1, col2 = st.columns(2)
    with col1:
        gender = list(df.gender.unique())
        gender_selections = st.multiselect("Membership type", gender)
    with col2:
        region_category = list(df.region_category.unique())
        region_selections = st.multiselect("Region", region_category)
    #KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        #info card of the Number of customers
        customers= df["Name"].nunique()
        theme_override = {'bgcolor': 'rgb(153, 255, 187)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-people-fill'}
        hc.info_card(title='Number of Customers', content=str(customers), theme_override=theme_override)
    with col2:
        #info card of the average transaction value
        avg_transaction_value=int(df["avg_transaction_value"].mean().round(0))
        theme_override = {'bgcolor': 'rgb(153, 255, 187)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-wallet'}
        hc.info_card(title='Average Transaction Value', content=str(avg_transaction_value), theme_override=theme_override)
    with col3:
        #info card of the average customer's age
        Average_age=int(df["age"].mean().round(0))
        theme_override = {'bgcolor': 'rgb(153, 255, 187)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-calendar-check'}
        hc.info_card(title='Customer Age Group', content=str(Average_age), theme_override=theme_override)
    with col4:
        #info card of the avg_time_spent_on_site
        avg_time_spent=int(df["avg_time_spent"].mean().round())
        theme_override = {'bgcolor': 'rgb(153, 255, 187)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-clock-history'}
        hc.info_card(title='Average Time Spent', content=str(avg_time_spent), theme_override=theme_override)






    #split to plot 2 side-by-side graphs
    col1, col2, col3 = st.columns(3)
    df = df[df['gender'].isin(gender_selections)]
    df = df[df['region_category'].isin(region_selections)]

    with col1:
        #Customer based on membership 
        df1 = df.groupby(['churn_risk_score', 'membership_category']).size().reset_index(name='counts')
        df1['churn_risk_score'] = df1['churn_risk_score'].astype(str)
        sorted1 = df1.sort_values(by='counts', ascending=False).reset_index()
        fig = go.Figure(px.histogram(df1, x='membership_category', y='counts',barmode='group',
            color_discrete_sequence= ['rgb(51, 204, 255)'],
             labels= {'counts':'Customers',
                     'membership_category': 'Membership type'}))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=400,
            title='Membership Categories Available',
            title_font_size=15)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                Most of the customers use {sorted1['membership_category'].loc[0]} as their membership type.
                </div>""",unsafe_allow_html = True)
    with col2:
        df2 = df.groupby(['churn_risk_score', 'region_category']).size().reset_index(name='counts')
        df2 = pd.pivot_table(df2, values=['counts'], columns=None, index=['region_category'], aggfunc='sum', sort=True)
        df2 = df2.reset_index()
        sorted2 = df2.sort_values(by='counts', ascending=False).reset_index()
        fig = go.Figure(px.pie(df2, values='counts', names='region_category',
            color_discrete_sequence= ['rgb(102, 255, 140)'],
             labels= {'region_category': 'Region',
                     'counts':'Number of Customers'}
            ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=400,
            title='Region Distribution',
            title_font_size=15)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                Most customer come from {sorted2['region_category'].loc[0]} Region.
                </div>""",unsafe_allow_html = True)

    with col3:
    #medium of operation
        df3= df.groupby(['medium_of_operation', 'churn_risk_score']).size().reset_index(name='counts')
        df3 = pd.pivot_table(df3, values=['counts'], columns=None, index=['medium_of_operation'], aggfunc='sum')
        df3 = df3.reset_index()
        sorted3 = df3.sort_values(by='counts', ascending=False).reset_index()
        fig = go.Figure(px.histogram(df3, x='medium_of_operation', y='counts', barmode='group',
            color_discrete_sequence= ['rgb(51, 204, 255)'],
             labels= {'medium_of_operation': 'Medium of Operation',
                     'counts':'Customers'}))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=400,
            title='Operations',
            title_font_size=15)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)
        st.caption(f"""
            <div>
                <div style="vertical-align:left;font-size:16px;padding-left:5px;padding-top:5px;margin-left:0em";>
                Customers use {sorted3['medium_of_operation'].loc[0]} and {sorted3['medium_of_operation'].loc[1]}as medium of operation
                </div>""",unsafe_allow_html = True)
    st.write("---")


    
    #split fot side-by-side plots
    col1, col2, col3 = st.columns(3)

    with col1:
    #plot4
        df10 = df.copy()
        df10['churn_risk_score'] = df10['churn_risk_score'].astype(str)
        fig = go.Figure(px.histogram(df10, x='avg_time_spent',
            color_discrete_sequence= ['rgb(51, 204, 255)'],
             labels= {'avg_time_spent': 'Average Time Spent'}
            ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=400,
            title='Average Time Spent',
            title_font_size=15)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)
    with col2:
        df15 = df.copy()
        df15['churn_risk_score'] = df15['churn_risk_score'].astype(str)
        fig = go.Figure(px.scatter(df15, x='avg_transaction_value', y='avg_time_spent', color='churn_risk_score',
            color_discrete_sequence= ['rgb(102, 255, 140)',"rgb(51, 204, 255)", "rgb(255, 179, 102)"],
             labels= {'churn_risk_score': 'Churn Score',
                 'counts':'Customers',
                 'avg_time_spent': 'Average Time Spent',
                 'avg_transaction_value': 'Average Transaction Value'}
        ))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=400,
            title='Time Spent amd Sales Value',
            title_font_size=15)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)
    with col3:
    #plot11
        df11 = df.groupby(['churn_risk_score']).size().reset_index(name='counts')
        df11['churn_risk_score'] = df11['churn_risk_score'].astype(str)
        fig = go.Figure(px.bar(df11, x='churn_risk_score', y='counts', color='churn_risk_score',
        color_discrete_sequence= ['rgb(102, 255, 140)',"rgb(51, 204, 255)", "rgb(255, 179, 102)"],
         labels= {'churn_risk_score': 'Churn Score',
                 'counts':'Customers'}))
        fig.update_layout(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), width=400,
        title='Churn Risk Score',
        title_font_size=15)
        fig.update_layout({ 'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)'})
        st.plotly_chart(fig)

if menu_id == 'Churn Prediction':
    
    df = pd.read_csv('Customer_Churn.csv')

    #seperate title and animation
    col1, col2= st.columns([2,2])
    with col1:
        st.title("Customer Churn Prediction")
        st.write("This app will allow you to predict if the customer will churn or not.")
        st.write("---")


    with col2:
        #animation13
        lottie_app= load_lottiefile("predict.json")
        st_lottie(lottie_app, height=360, width=650)

    
    
    #specifying numerical and categorical features
    numerical_features = ['age', 'days_since_last_login', 'avg_time_spent',
       'avg_transaction_value', 'avg_frequency_login_days', 'points_in_wallet',
       'Tenure (months)']

    df_numerical = df[numerical_features]

    categorical_features =['gender', 'region_category', 'membership_category',
       'joined_through_referral', 'preferred_offer_types',
       'medium_of_operation', 'internet_option', 'used_special_discount',
       'offer_application_preference', 'past_complaint', 'complaint_status',
       'feedback']
    df_categorical = df[categorical_features]

       # Features
    X = pd.concat([df_categorical, df_numerical], axis = 1)

       # Target Variable
    y = df['churn_risk_score']

    #defining function to create inputs that will give a prediction
    def user_input_features():

        col3, col4, col5= st.columns(3)
        with col3:
           Age = st.number_input("Age", value=10, min_value=10)
           Last_login = st.number_input('Days since the last time logged in', value=1, min_value=1)
           Time_spent = st.number_input("Average time spent ", value=1, min_value=1)
           Transaction = st.number_input("Average value of transactions ", value=1, min_value=1)
           Frequency = st.number_input("Average times spent", value=1, min_value=1)
           Points = st.number_input("Points in wallet", value=0, min_value=-1000)
           Feedback= st.selectbox('Customer feedback', ('Poor Product Quality',
           'No reason specified', 'Too many ads', 'Poor Website', 'Poor Customer Service', 'Reasonable Price',
           'User Friendly Website', 'Products always in Stock', 'Quality Customer Care'))


        
           
        with col4:
            Complaint= st.selectbox('Past complaints', ('Yes', 'No'))
            Comp_stat= st.selectbox('Status of the complaint', ('Not Applicable', 'Unsolved', 'Solved',
           'Solved in Follow-up', 'No Information Available'))
            Gender= st.selectbox('Gender', ('Female', 'Male'))
            Region= st.selectbox('Region', ('Town', 'City', 'Village'))
            Membership= st.selectbox('Membership category', ('Basic Membership', 'No Membership',
           'Gold Membership', 'Silver Membership', 'Premium Membership', 'Platinum Membership'))
            Tenure = st.number_input("Tenure", value=0, min_value=0)

        with col5:
           Referral = st.selectbox('Offer type', ('Yes', 'No', 'Maybe'))
           Offer= st.selectbox("Order type", ('Gift Vouchers/Coupons',
           'Credit/Debit Card Offers', 'Without Offers'))
           Medium = st.selectbox("Medium of operation", ('Desktop', 'Smartphone', 'Not Specified', 'Both'))
           Internet = st.selectbox("Internet option", ('Wi-Fi', 'Mobile_Data', 'Fiber_Optic'))
           Discount = st.selectbox('Use of a special discount?', ('Yes', 'No'))
           Prefer_offer= st.selectbox('Offer preference?', ('Yes', 'No'))
           
           
           data = {'age': Age, 'days_since_last_login': Last_login, 'avg_time_spent': Time_spent,
           'avg_transaction_value': Transaction, 'avg_frequency_login_days': Frequency, 'points_in_wallet': Points,
           'Tenure (months)': Tenure, 'gender': Gender, 'region_category':Region,
            'membership_category': Membership, 'joined_through_referral': Referral,
            'preferred_offer_types': Offer, 'medium_of_operation': Medium, 'internet_option': Internet,
            'used_special_discount': Discount, 'offer_application_preference': Prefer_offer,
            'past_complaint': Complaint, 'complaint_status': Comp_stat, 'feedback': Feedback}

           features= pd.DataFrame(data, index=[0])
           return features
           

    df1= user_input_features()

    #encoding categorical variable
    encoderlabel = LabelEncoder()
    y = encoderlabel.fit_transform(y)

    #pipeline for all necessary transformations
    cat_pipeline= Pipeline(steps=[
            ('impute', SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 'None')),
            ('ohe', OneHotEncoder(handle_unknown = 'ignore'))
            ])

    num_pipeline = Pipeline(steps=[

            ('impute', SimpleImputer(missing_values = np.nan, strategy='mean')),
            ('outlier',RobustScaler())
            ])

    column_transformer= ColumnTransformer(transformers=[
            ('ohe', cat_pipeline, categorical_features),
            ('impute', num_pipeline, numerical_features)
            ], remainder='drop')

    #chose best model based on previous trials
    model = RandomForestClassifier(class_weight='balanced')

    pipeline_model = Pipeline(steps = [('transformer', column_transformer),
                             ('model', model)])

    #train the model
    pipeline_model.fit(X, y)

    #predicting the data
    prediction = pipeline_model.predict(df1)

    m = st.markdown("""
        <style>
            div.stButton > button:first-child {
            color: rgb(255, 102, 102);
            background-color: rgb(153, 255, 187);
            }
        </style>""", unsafe_allow_html=True)
    submit = st.button('Predict')

    if submit:
        st.subheader(f'The Customer is likely to churn by {str(prediction)}')
        st.write('---')
        
        
        
if menu_id == 'Profiling':
    name = pd.Series(df['Name'])
    selection = st.selectbox('Select a Customer', name)
    df20 = df.loc[df['Name'] == selection]
    #seperation between graphs
    col1,col2, col3 = st.columns([2,1,1])

    with col1:
        

        for i in df20['gender']:
            if i == 'Female':
                #if the customer is female, return a female animated image
                lottie_girl= load_lottiefile("woman.json")
                st_lottie(lottie_girl, height=550, width=600)

            else:
                #if the customer is male, return a male animated image
                lottie_guy= load_lottiefile("man.json")
                st_lottie(lottie_guy, height=550, width=600)


        churn = df20['churn_risk_score'].values[0]

        feedback = df20['feedback'].values[0]
        theme_override = {'bgcolor': 'rgb(255, 102, 102)','title_color': 'white','content_color': 'white','progress_color': 'rgb(136,204,238)',
        }

    with col2:

        #info card of the customer's age
        age = df20['age'].values[0]
        theme_override = {'bgcolor': 'rgb(26, 255, 140)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-calendar-check'}
        hc.info_card(title='Age', content=str(age), theme_override=theme_override)


        #info card of the customer's region
        region = df20['region_category'].values[0]
        theme_override = {'bgcolor': 'rgb(26, 255, 140)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-compass'}
        hc.info_card(title='Region', content=str(region), theme_override=theme_override)

        #info card of the customer's churn risk
        Churn = df20['churn_risk_score'].values[0]
        theme_override = {'bgcolor': 'rgb(26, 255, 140)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-exclamation-triangle'}
        hc.info_card(title='Churn Score', content=str(Churn), theme_override=theme_override)



    with col3:

        #info card of the customer's internet option
        internet_option = df20['internet_option'].values[0]
        theme_override = {'bgcolor': 'rgb(26, 255, 140)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-router'}
        hc.info_card(title='internet_option', content=str(internet_option), theme_override=theme_override)

        #info card of the customer's membership category
        membership = df20['membership_category'].values[0]
        theme_override = {'bgcolor': 'rgb(26, 255, 140)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-patch-check'}
        hc.info_card(title='Membership', content=str(membership), theme_override=theme_override)

        #info card of the customer's average frequency login days
        frequency = df20['avg_frequency_login_days'].values[0]
        theme_override = {'bgcolor': 'rgb(26, 255, 140)','title_color': 'white','content_color': 'white',
        'icon_color': 'white', 'icon': 'bi bi-bag-check'}
        hc.info_card(title='Frequency Login', content=str(frequency) + ' Days', theme_override=theme_override)


#BREAK
webapp_style()