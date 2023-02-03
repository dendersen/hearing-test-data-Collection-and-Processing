import streamlit as st
import pandas as pd
import csv
import Main 
IDInformation = pd.read_csv('Data\ID_collection.csv')

def save_data():
  '''
  Saves the data to to the file named ID_collection.csv and returns the current ID number
  '''
  
  if (IDInformation['ID'].max() != IDInformation['ID'].max()):
    newID = 0
  else:
    newID = 1+IDInformation['ID'].max()
  
  if Gender == "Woman":
    G = '1'
  else:
    G = '0'
  
  if HearingLoss == "do not have hearingLoss":
    H = '0'
  else:
    H = '1'
  #we create a dataframe we want to save
  data = {'ID': [newID],
          'Gender': [G],
          'Age': [Age],
          'HearingLoss': [H],
          'HeadphoneTime': [HeadphoneTime],
          'PlaceOfTest': [PlaceOfTest],
          'Name': [Name],}
  df = pd.DataFrame(data)
  #we can now append the data to the csv-file
  df.to_csv('Data\ID_collection.csv', mode='a', index=False, header=False)
  return newID

st.title('The bedst hearing test online:')

col1, col2 = st.columns(2) #Splits the screan into two halves
with col1:
  st.header('Indput data here')
  #User values are given so they can be saved
  Name = st.text_input('Name')
  Gender = st.selectbox('Gender',('Man','Woman'))
  Age = st.number_input('Age',min_value=0,max_value=150)
  HearingLoss = st.selectbox('HearingLoss',('have been diagnosed with hearingLoss','do not have hearingLoss'))
  HeadphoneTime = st.number_input('HeadphoneTime in hours/day',min_value=0,max_value=24)
  PlaceOfTest = st.text_input('PlaceOfTest')
  
  st.header('Tester have to do the rest :smile:')
  
  Start = st.number_input('Manual input 1',value=200.0)
  Slut = st.number_input('Manual input 2',value=17000.0)
  # minMaxFrequency = st.slider(
  #   'Select a range of frequency',
  #   0.0, 30000.0, value=(Start,Slut))
  
  # st.write('Values:', minMaxFrequency)
  
  numberOfTones = st.number_input('Indput number of tones', min_value=2, max_value=100)
  
  Starter = st.button('Start test')
  # The tests starts when button is clicked
  if Starter: 
    st.write('Test has been started :smile:')
    id = save_data()
    Main.runTestSequence(id,Start,Slut,numberOfTones)
    st.write('Test done :smile:')


with col2:
  st.header('See output')
  st.write('Currently getting data from: ',Name)
  st.write('The gender is: ',Gender)
  st.write('The age is: ',Age)
  if Age == 69: # funnies
    st.write('Nice :smile:')
  
  st.write('The one in question ',HearingLoss)
  st.write(Name,' uses the headphone: ',HeadphoneTime,'hour/day')
  st.write('The test is taken at: ',PlaceOfTest)
  st.write(IDInformation)
  
  st.button('Refresh data')



