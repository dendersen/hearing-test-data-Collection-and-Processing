import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import Visualization.TreatedData as VT

#reads data
data1 = pd.read_csv('Data\editResultStorage.csv')
data2 = pd.read_csv('Data\ID_collection.csv')
#make two lists of unique ears and plots
availablePlots = data1['ID'].unique()
availableEars = data1['Out'].unique()

st.header('View test results here')

update = st.button('Update data')
if update:
  VT.Clear_Data()
  VT.Change_Dataform()

#Choose id to show
idChosen = st.selectbox('Select person to show',availablePlots)
#Choose ear to show
earList = ['no ear', 'left ear', 'right ear', 'both ears']
ears =[earList[i] for i in availableEars]
earChosen = st.selectbox('Results of sound played to ',ears)
#we plot the data


st.header('Answers are shown with colors')
st.write(f'<h1 style="color:#FF0000;font-size:18px;">{"Red = cannot hear (0)"}</h1>', unsafe_allow_html=True)
st.write(f'<h1 style="color:#A020F0;font-size:18px;">{"Purple = Answer in left ear (1)"}</h1>', unsafe_allow_html=True)
st.write(f'<h1 style="color:#00FF00;font-size:18px;">{"Green = Sound in right ear (2)"}</h1>', unsafe_allow_html=True)
st.write(f'<h1 style="color:#0000FF;font-size:18px;">{"Blue = both ears heard the sound (3)"}</h1>', unsafe_allow_html=True)
colorList = ['red', 'purple', 'darkgreen', 'blue']

dff = data1[data1['ID'] == idChosen]  
dfff = dff[dff['Out'] == earList.index(earChosen)]

idData = data2[data2['ID'] == idChosen]

fig, ax = plt.subplots()
ax.scatter(dfff['Frekvens'], dfff["AnswerTime"],c=[colorList[i] for i in dfff["Response"]])
ax.set_title('| Data of ID: '+str(idChosen)+' | Name: '+ str(idData['Name'].item()) +' | Currently showing '+ str(earChosen)+' |', color="black")
ax.set_xlabel('Frequency')
ax.set_ylabel('Answer time in ms')
st.pyplot(fig)

st.header('Genneral data analysis')

#Evalf correct responses
index = dff.index
responseEval = pd.DataFrame(columns = ['ear', 'answerTjek'])
for i in range(len(dff)):
  if dff['Out'][index[i]] == dff['Response'][index[i]]:
    responseEval.loc[len(responseEval)] = [dff['Out'][index[i]], 1] #if answer is correct, then i'll set the value to 1
  else:
    responseEval.loc[len(responseEval)] = [dff['Out'][index[i]], 0] 


noEarAnswer = responseEval[responseEval['ear'] == 0]['answerTjek']
if noEarAnswer.value_counts()[1] == noEarAnswer.count():
  if noEarAnswer[noEarAnswer.first_valid_index()] == 1:
    answerListNoEar = ['Correct']
  else:
    answerListNoEar = ['Not correct']
else:
  answerListNoEar = ['Correct', 'Not Correct']


leftEarAnswer = responseEval[responseEval['ear'] == 1]['answerTjek']
if leftEarAnswer.value_counts()[1] == leftEarAnswer.count():
  if leftEarAnswer[leftEarAnswer.first_valid_index()] == 1:
    answerListLeftEar = ['Correct']
  else:
    answerListLeftEar = ['Not correct']
else:
  answerListLeftEar = ['Correct', 'Not Correct']

rightEarAnswer = responseEval[responseEval['ear'] == 2]['answerTjek']
if rightEarAnswer.value_counts()[1] == rightEarAnswer.count():
  if rightEarAnswer[rightEarAnswer.first_valid_index()] == 1:
    answerListRightEar = ['Correct']
  else:
    answerListRightEar = ['Not correct']
else:
  answerListRightEar = ['Correct', 'Not Correct']

bothEarAnswer = responseEval[responseEval['ear'] == 3]['answerTjek']
if bothEarAnswer.value_counts()[1] == bothEarAnswer.count():
  if bothEarAnswer[bothEarAnswer.first_valid_index()] == 1:
    answerListBothEar = ['Correct']
  else:
    answerListBothEar = ['Not correct']
else:
  answerListBothEar = ['Correct', 'Not Correct']

st.write(f'<h1 style="color:#00000;font-size:22px;">{"Correctness of answers"}</h1>', unsafe_allow_html=True)
fig, ax = plt.subplots(nrows=4, sharex=False, sharey=False, figsize = (10,14))

ax[0].barh(answerListNoEar,noEarAnswer.value_counts(), color = 'red')
totals = []
for i in ax[0].patches:
  totals.append(i.get_width())
total = sum(totals)
for i in ax[0].patches:
  ax[0].text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12, color='black')
ax[0].set_title('No ear', color = 'red')


ax[1].barh(answerListLeftEar,leftEarAnswer.value_counts(),color = 'purple')
totals = []
for i in ax[1].patches:
  totals.append(i.get_width())
total = sum(totals)
for i in ax[1].patches:
  ax[1].text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12, color='black')
ax[1].set_title('\n Left ear',color = 'purple')

ax[2].barh(answerListRightEar,rightEarAnswer.value_counts(),color = 'darkgreen')
totals = []
for i in ax[2].patches:
  totals.append(i.get_width())
total = sum(totals)
for i in ax[2].patches:
  ax[2].text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12, color='black')
ax[2].set_title('\n Right ear',color = 'darkgreen')

ax[3].barh(answerListBothEar,bothEarAnswer.value_counts(),color = 'blue')
totals = []
for i in ax[3].patches:
  totals.append(i.get_width())
total = sum(totals)
for i in ax[3].patches:
  ax[3].text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12, color='black')
ax[3].set_title('\n Both ears',color = 'blue')
st.pyplot(fig)