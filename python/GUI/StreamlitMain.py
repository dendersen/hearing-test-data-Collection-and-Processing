import streamlit as st
import pandas as pd
import csv
table = pd.read_csv('ID_collection.csv')

def save_data():
  data = {'ID': [1+table['ID'].max()],
          'Name': [Name],
          'Gender': [Gender],
          'Age': [Age],
          'HearingLoss': [HearingLoss],
          'HeadphoneTime': [HeadphoneTime],
          'PlaceOfTest': [PlaceOfTest]}
  df = pd.DataFrame(data)
  df.to_csv('ID_collection.csv', mode='a', index=False, header=False)

st.title('The bedst hearing test online:')

col1, col2 = st.columns(2)
with col1:
  st.header('Indput data here')
  Name = st.text_input('Name')
  Gender = st.selectbox('Gender',('Man','Whoman'))
  Age = st.number_input('Age',min_value=0,max_value=150)
  HearingLoss = st.selectbox('HearingLoss',('have been diagnosed with hearingLoss','do not have hearingLoss'))
  HeadphoneTime = st.number_input('HeadphoneTime in hours/day',min_value=0,max_value=24)
  PlaceOfTest = st.text_input('PlaceOfTest')
  st.header('Tester have to do the rest :smile:')
  Start = st.number_input('Manual input 1',value=2500.0)
  Slut = st.number_input('Manual input 2',value=17500.0)
  values = st.slider(
    'Select a range of frequency',
    0.0, 20000.0, value=(Start,Slut))
  st.write('Values:', values)
  Starter = st.button('Start test')
  if Starter:
    st.write('Test has been started :smile:')
    save_data()
with col2:
  st.header('See output')
  st.write('Currently getting data from: ',Name)
  st.write('The gender is: ',Gender)
  st.write('The age is: ',Age)
  if Age == 69:
    st.write('Nice :smile:')
  st.write('The one in question ',HearingLoss)
  st.write(Name,' uses the headphone: ',HeadphoneTime,'hour/day')
  st.write('The test is taken at: ',PlaceOfTest)
  st.write(table)
  st.button('Refresh data')



