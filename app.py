# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:09:52 2023

@author: l.baltaji
"""
import datetime
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

from PIL import Image
# Loading Image using PIL
im = Image.open('logo-removebg-preview2.png')

# Adding Image to web app
st.set_page_config(page_title="Anomaly Detector App", page_icon = im)

# Load the saved autoencoder model
with open('autoencoder.pkl', 'rb') as model_file:
    loaded_autoencoder = pickle.load(model_file)
with open('scaler.pkl', 'rb') as model_file3:
    scaler = pickle.load(model_file3)

def welcome():
    return "Welcome All"



def detect_anomalies(input_data):
    
    # Predict anomalies using the loaded autoencoder model
    prediction = loaded_autoencoder.predict(input_data)

    # Calculate the mean absolute error
    prediction_loss = np.mean(np.abs(prediction - input_data), axis = 1)
    
    # Extract the single prediction loss value
    prediction_loss_scalar = prediction_loss[0]

    # Define a threshold for anomaly detection
    anomaly_threshold = 0.2  

    # Display the prediction and anomaly status
    if prediction_loss_scalar > anomaly_threshold:
        result = 'Prediction: Suspected to be Anomalous'
    else:
        result = 'Prediction: Not Anomalous'
    return result

def detect_outside_working_time(df):
    # Define working hours
    start_working_hour = 8
    end_working_hour = 18
    # Define non-working day (Friday)
    non_working_day = 4

    def is_outside_working_time(row):
        if row['weekday'] != non_working_day:
            if start_working_hour <= row['hour'] < end_working_hour:
                return 0
        return 1
    
    outside_working_time = df.apply(is_outside_working_time, axis=1)
    return outside_working_time

# Extract binary field being 1 if the date difference is negative and 0 otherwise
def detect_negative_date_difference(df):
    negative_date_difference = (df['date_difference'] < 0).astype(int)
    return negative_date_difference

def unitprice_avgcost_ratio(df):
    unitprice_avgcost_ratio = df['UnitPrice']/(df['AVGCost']+1)
    return unitprice_avgcost_ratio

def detect_localvaluerate_outliers(df):
    outlier_localvaluerate = (df['LocalValueRate']>35).astype(int)
    return outlier_localvaluerate

def detect_negative_quantity_or_valueamount(df):
    negative_quantity_or_valueamount = ((df['Quantity'] <= 0) | (df['ValueAmount'] <= 0)).astype(int)
    return negative_quantity_or_valueamount

def add_extra_features(df):
    outside_working_time = detect_outside_working_time(df)
    negative_date_difference = detect_negative_date_difference(df)
    unitprice_avgcost_ratio_val = unitprice_avgcost_ratio(df)
    localvaluerate_outliers = detect_localvaluerate_outliers(df)
    negative_quantity_or_valueamount = detect_negative_quantity_or_valueamount(df)
    
    # Create a new DataFrame with the additional features
    new_columns = ['outside_working_time', 'negative_date_difference','unitprice_avgcost_ratio', 'localvaluerate_outliers', 'negative_quantity_valueamount']
    
    extra_features_df = pd.DataFrame({
        'outside_working_time' : outside_working_time,
        'negative_date_difference': negative_date_difference,
        'unitprice_avgcost_ratio': unitprice_avgcost_ratio_val,
        'localvaluerate_outliers': localvaluerate_outliers,
        'negative_quantity_valueamount': negative_quantity_or_valueamount
    }, columns=new_columns)
    
    # Concatenate the new DataFrame with the original DataFrame
    combined_df = pd.concat([df, extra_features_df], axis=1)
    
    return combined_df

features_to_select = ['Quantity', 'TicketCharges', 'ValueAmount', 'LocalValueRate', 'LocalEquivalent', 
                        'outside_working_time', 'negative_date_difference', 'unitprice_avgcost_ratio', 
                      'localvaluerate_outliers','negative_quantity_valueamount']

numerical_cols = ['Quantity','UnitPrice','AVGCost','ExchangeFees', 'BrokerageFees','TicketCharges', 'SettlementFees', 'OtherFees',
                  'ValueAmount', 'LocalValueRate', 'LocalEquivalent', 'hour', 'weekday', 'date_difference', 
                  'unitprice_avgcost_ratio']


st.title('Anomaly Detection Application')
st.image(im, caption='IDS Fintech Logo')
app_mode = st.sidebar.selectbox('Select Page', ['About', 'Tableau Dashboards', 'Anomaly Detector Using Auto-Encoder'])
if app_mode == 'About':
    st.title('Background about the Company and Project')
    st.subheader('What is IDS Fintech?')
    st.write("Integrated Digital Systems Fintech (IDS Fintech) is a leading provider of cutting-edge Financial Software Solutions in the MENA market. Specializing in portfolio management and trading solutions, IDS Fintech empowers financial companies, including investment banks, brokerage firms, funds and family offices, with state-of-the-art tools for effective management, automation, and monitoring of their investment processes. ")
    st.write('At the heart of IDS Fintech’s comprehensive suite of solutions lies Vestio, an advanced asset management system that revolutionizes the way financial institutions handle their investment operations. Vestio brings together a powerful combination of modelling, benchmarking, rebalancing, risk calculators, and ad-hoc compliance tools, providing a platform that assures seamless integration with trading engines, such as ROR, and depository containers, including accounting systems.')
    st.write('The primary goal of this collaborative project was to develop and implement an anomaly detection model tailored to IDS Fintech’s transactional financial data in order to enhance their security and operational efficiency. ')
    data = pd.read_excel(r'C:\Users\l.baltaji\Desktop\Capstone Project\Methodology and Source Code\Tickets Table Plus Synthesized Anomalies.xlsx')
    st.markdown('Dataset Used: ')
    st.write(data.head())
        
    st.write("The anomaly detector presented in the third page is only a prototype which still requires more optimization and tuning in order to be used at IDS Fintech and implemented in Vestio, their asset management system.")

if app_mode =='Tableau Dashboards':
    st.title('Tableau Dashboards')
    st.subheader('Dahboard 1: ')
    components.html(
        """
        <div class='tableauPlaceholder' id='viz1694470569140' style='position: relative'><noscript><a href='#'><img alt='Dashboard 1 - Data Exploration ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard1-DataExploration&#47;Dashboard1-DataExploration&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Dashboard1-DataExploration&#47;Dashboard1-DataExploration' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard1-DataExploration&#47;Dashboard1-DataExploration&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1694470569140');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1200px';vizElement.style.height='1127px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1200px';vizElement.style.height='1127px';} else { vizElement.style.width='100%';vizElement.style.height='1877px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
        """,
        height=1200,
        width = 2000)
    st.subheader('Dashboard 2: ')
    components.html(
        """
       <div class='tableauPlaceholder' id='viz1694471014626' style='position: relative'><noscript><a href='#'><img alt='Dashboard 2 - Pearson&#39;s Correlation between Anomaly Types and Features ' src='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard2-PearsonsCorrelationbetweenAnomalyTypesandFeatures&#47;Dashboard2-PearsonsCorrelationbetweenAnomalyTypesandFeatures&#47;1_rss.png' style='border: none' /></a></noscript><object class='tableauViz'  style='display:none;'><param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' /> <param name='embed_code_version' value='3' /> <param name='site_root' value='' /><param name='name' value='Dashboard2-PearsonsCorrelationbetweenAnomalyTypesandFeatures&#47;Dashboard2-PearsonsCorrelationbetweenAnomalyTypesandFeatures' /><param name='tabs' value='no' /><param name='toolbar' value='yes' /><param name='static_image' value='https:&#47;&#47;public.tableau.com&#47;static&#47;images&#47;Da&#47;Dashboard2-PearsonsCorrelationbetweenAnomalyTypesandFeatures&#47;Dashboard2-PearsonsCorrelationbetweenAnomalyTypesandFeatures&#47;1.png' /> <param name='animate_transition' value='yes' /><param name='display_static_image' value='yes' /><param name='display_spinner' value='yes' /><param name='display_overlay' value='yes' /><param name='display_count' value='yes' /><param name='language' value='en-US' /></object></div>                <script type='text/javascript'>                    var divElement = document.getElementById('viz1694471014626');                    var vizElement = divElement.getElementsByTagName('object')[0];                    if ( divElement.offsetWidth > 800 ) { vizElement.style.width='1200px';vizElement.style.height='1027px';} else if ( divElement.offsetWidth > 500 ) { vizElement.style.width='1200px';vizElement.style.height='1027px';} else { vizElement.style.width='100%';vizElement.style.height='2127px';}                     var scriptElement = document.createElement('script');                    scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';                    vizElement.parentNode.insertBefore(scriptElement, vizElement);                </script>
        """,
        height=1300,
        width = 2000)
if app_mode =='Anomaly Detector Using Auto-Encoder':
    st.title('Anomaly Detector Using Auto-Encoder Deep Learning Model')
    st.subheader('What is an Auto-Encoder?')
    st.write('An Auto-Encoder is a unsupervised deep learning model which consists of an encoder network that compresses only the normal input data into a lower-dimensional latent space and a decoder network that reconstructs the original normal input from the compressed representation, as seen in the figure below. ') 
    st.image('auto_encoder_architecture.jpg', caption = 'Basic Architecture of an AutoEncoder Model. Adapted from Sublime and Kalinicheva, 2019')
    st.write("This anomaly detector is only a prototype which still requires more optimization and tuning in order to be used at IDS Fintech and implemented in Vestio, their asset management system.")
    st.subheader('Try it yourself:')
    input_features = {}
    for feature_numerical in ['TypeId', 'ClientId', 'ProductId', 'BrokerId', 'Quantity', 
                              'UnitPrice', 'AVGCost','ExchangeFees','BrokerageFees', 'TicketCharges', 'SettlementFees',
                              'OtherFees', 'ValueAmount', 'LocalValueRate','LocalEquivalent']:
        input_features[feature_numerical] = st.number_input(f'Enter {feature_numerical}:', min_value= -1000.0, value=0.00)
    for feature_date in ['CashSettlementDate',	'ShareSettlementDate']:
        input_features[feature_date] = st.date_input(f'Enter {feature_date}:', value=datetime.datetime(2023, 1, 1))
    for feature_datetime in ['CreationDate']:
        creation_date = st.date_input(f'Enter {feature_datetime}', value = datetime.datetime(2023, 1, 1))
        creation_time = st.time_input('Enter CreationTime', value = datetime.time(8,45))
        creation_datetime = datetime.datetime.combine(creation_date, creation_time)
        input_features[feature_datetime] = creation_datetime
        
    input_data = pd.DataFrame(input_features, index=[0])
    input_data['hour'] = input_data['CreationDate'].dt.hour
    input_data['weekday'] = input_data['CreationDate'].dt.weekday
    input_data['date_difference'] = (input_data['CashSettlementDate'] - input_data['ShareSettlementDate']).dt.days
    input_data = add_extra_features(input_data)
    input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    input_data = input_data[features_to_select]

    result=""
    
    if st.button("Predict", key="predict_button", help="Click to predict anomalies"):
        result = detect_anomalies(input_data)
        if result == 'Prediction: Not Anomalous':
            st.balloons()
            st.success(result)
        else:
            st.error(result)
    


