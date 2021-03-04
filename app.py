#!pip install streamlit
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image



st.markdown("<h1 style='text-align: center; color: #d7385e ;'><strong><u>Predict Blood Donation for Future</u></strong></h1>", unsafe_allow_html=True)

image = Image.open('donate.jpg')
st.image(image,width=700)

st.sidebar.markdown("<h1 style='text-align: center; color:#d7385e ;'><strong><u>Specify Input Parameters</u></strong></h1>", unsafe_allow_html=True)
    
st.markdown("Forecasting blood supply is a serious and recurrent problem for blood collection managers: in January 2019, Nationwide, they saw 27,000 fewer blood donations over the holidays than they see at other times of the year.")
st.markdown("Understanding the Parameters -")
st.markdown("(Recency - Months since the Last donation)")
st.markdown("(Frequency - Total Number of Donations)")
st.markdown("(Monetary - Total Blood Donated in C.C.)")
st.markdown("(Time - Months Since the First Donation)")
st.markdown("Target - (1 stands for Donating Blood, 0 stands for Not Donating Blood)")




def user_input_features():
    Recency  = st.sidebar.slider('Recency', 0, 74)
    Frequency= st.sidebar.slider('Frequency', 1,43)
    Monetary = st.sidebar.slider('Monetary', 250,12500)
    Time = st.sidebar.slider('Time', 2,98)

     
    data = {'Recency (months)': Recency  ,
           'Frequency (times)': Frequency,
           'Monetary (c.c. blood)': Monetary,
           'Time (months)':Time}
           
           
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.write(df)

trans = pd.read_csv('transfusion.csv')

X_train,X_test,y_train,y_test = train_test_split(trans.drop(columns=['target']), trans['target'].values,test_size=0.2, random_state=24)

clf = LogisticRegression()
clf.fit(X_train, y_train)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)


st.subheader('Prediction')
st.write(trans.target[prediction])

st.subheader('Prediction Probability')
st.write(prediction_proba)
