import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

#reads data
data1 = pd.read_csv('Data\editResultStorage.csv')
data2 = pd.read_csv('Data\ID_collection.csv')
#make two lists of unique ears and plots
availablePlots = data1['ID'].unique()
availableEars = data1['Out'].unique()

st.header('View test results here')

#Choose id to show
idChosen = st.selectbox('Select person to show',availablePlots)
#Choose ear to show
earList = ['no ear', 'left ear', 'right ear', 'both ears']
ears =[earList[i] for i in availableEars]
earChosen = st.selectbox('Select ear to show',ears)
#we plot the data


st.header('Answers are shown with colors')
st.write(f'<h1 style="color:#FFFF00;font-size:18px;">{"Yellow = cannot hear (0)"}</h1>', unsafe_allow_html=True)
st.write(f'<h1 style="color:#FF0000;font-size:18px;">{"Red = Answer in left ear (1)"}</h1>', unsafe_allow_html=True)
st.write(f'<h1 style="color:#00FF00;font-size:18px;">{"Green = Sound in right ear (2)"}</h1>', unsafe_allow_html=True)
st.write(f'<h1 style="color:#FFC0CB;font-size:18px;">{"Pink = both ears heard the sound (3)"}</h1>', unsafe_allow_html=True)
colorList = ['yellow', 'red', 'green', 'pink']

dff = data1[data1['ID'] == idChosen]  
dfff = dff[dff['Out'] == earList.index(earChosen)]

idData = data2[data2['ID'] == idChosen]

fig, ax = plt.subplots()
ax.scatter(dfff['Frekvens'], dfff["AnswerTime"],c=[colorList[i] for i in dfff["Response"]])
ax.set_title('| Data of ID: '+str(idChosen)+' | Name: '+ str(idData['Name'].item()) +' | Currently showing '+ str(earChosen)+' |', color="red")
ax.set_xlabel('Frequency')
ax.set_ylabel('Answer time in ms')
ax.set_facecolor('darkblue')
st.pyplot(fig)