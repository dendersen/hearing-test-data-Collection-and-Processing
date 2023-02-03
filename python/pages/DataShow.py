import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import Visualization.TreatedData as VT
import seaborn as sns

#reads data
data1 = pd.read_csv('Data\editResultStorage.csv')
data2 = pd.read_csv('Data\ID_collection.csv')
#make two lists of unique ears and plots
availablePlots = data1['ID'].unique()
availableEars = data1['Out'].unique()

update = st.button('Update data')
if update:
  VT.Clear_Data()
  VT.Change_Dataform()

whatToShow = st.selectbox('How to show data',('Over all stats','Individual results'))
if whatToShow == 'Over all stats':
  st.header('Overall stats')
  
  #we make a list of unique frequencies and sort it
  listOfFreaquency = data1['Frekvens'].sort_values().unique()
  
  #plot of frequency and how many test we have with it 
  fig, ax = plt.subplots(ncols=1,nrows=1)
  #we count how many times a specific frequency is used and sort it
  freaquenzyCount = data1['Frekvens'].value_counts().sort_index()
  ax.plot(listOfFreaquency[0:],freaquenzyCount[freaquenzyCount.index[0]:])
  ax.set_title('Amount of test at given frequency')
  ax.set_ylabel('Times frequency is used')
  ax.set_xlabel('Frequency')
  st.pyplot(fig)
  
  #plot of correct answers against the actual answer
  fig, ax = plt.subplots(1, 2,  sharex=False, sharey=False, figsize=(15,10))
  fig.suptitle('Correct answer'+"   vs   "+'Response', fontsize=20)
  ### count
  ax[0].title.set_text('count')
  order = data1.groupby(['Out','Response'])['Response'].count()
  order = order.reset_index(name = 'count')
  sns.barplot(data=order, x='Out', y='count', hue='Response', ax=ax[0])
  ax[0].grid(True)
  ### percentage
  ax[1].title.set_text('percentage')
  a = data1.groupby('Out')['Response'].count().reset_index()
  a = a.rename(columns={'Response':"tot"})
  b = data1.groupby(['Out','Response'])['Response'].count()
  b = b.reset_index(name = 'count')
  b = b.rename(columns={'count':0}).reset_index()
  b = b.merge(a, how="left")
  b["%"] = b[0] / b["tot"] *100
  sns.barplot(x='Out', y="%", hue='Response', data=b, ax=ax[1]).get_legend().remove()
  ax[1].grid(True)
  ### fix figure
  st.pyplot(fig)
  
  #we make a list of how many time the correct answer is given to a specific frequency
  listOfAccuracy = [1 if data1['Out'][i] == data1['Response'][i] else 0 for i in range(len(data1))]
  
  listOfFreaquencyAccuracy = pd.DataFrame(0,index=range(len(listOfFreaquency)),columns=list('AB'))
  for i in range(len(data1)):
    CurrentFreaquenzy = data1['Frekvens'][i]
    indexOfFreaquenzy = np.where(listOfFreaquency == CurrentFreaquenzy)
    if listOfAccuracy[i] == 0:
      listOfFreaquencyAccuracy.loc[indexOfFreaquenzy[0][0],'A'] = listOfFreaquencyAccuracy['A'][indexOfFreaquenzy[0][0]]+1
    else:
      listOfFreaquencyAccuracy.loc[indexOfFreaquenzy[0][0],'B'] = listOfFreaquencyAccuracy['B'][indexOfFreaquenzy[0][0]]+1
  
  for i in range(len(listOfFreaquencyAccuracy)):
    count = listOfFreaquencyAccuracy['A'][i]+listOfFreaquencyAccuracy['B'][i]
    listOfFreaquencyAccuracy.loc[i,'A'] = listOfFreaquencyAccuracy['A'][i]/count*100
    listOfFreaquencyAccuracy.loc[i,'B'] = listOfFreaquencyAccuracy['B'][i]/count*100
  
  fig, ax = plt.subplots(ncols=1,nrows=7,figsize = (10,30))
  ax[0].set_title('Accuracy of freaquencies')
  
  for i in range(7):
    ax[i].plot(listOfFreaquency[round(len(listOfFreaquency)*i/7):round(len(listOfFreaquency)*(i+1)/7)],listOfFreaquencyAccuracy['B'][round(len(listOfFreaquency)*i/7):round(len(listOfFreaquency)*(i+1)/7)])
    ax[i].set_ylabel('Accuracy in %')
    ax[i].set_xlabel('Frequency')
  st.pyplot(fig)
else:
  st.header('Individual results here')
  
  #Choose id to show
  idChosen = st.selectbox('Select person to show',availablePlots)
  #Choose ear to show
  earList = ['no ear', 'left ear', 'right ear', 'both ears']
  ears =[earList[i] for i in availableEars]
  earChosen = st.selectbox('Results of sound played to ',ears)
  #we plot the data
  
  st.header('Answers are shown with colors')
  st.write(f'<h1 style="color:#0000FF;font-size:18px;">{"Blue = cannot hear (0)"}</h1>', unsafe_allow_html=True)
  st.write(f'<h1 style="color:#FFA500;font-size:18px;">{"Orange = Answer in left ear (1)"}</h1>', unsafe_allow_html=True)
  st.write(f'<h1 style="color:#00FF00;font-size:18px;">{"Green = Sound in right ear (2)"}</h1>', unsafe_allow_html=True)
  st.write(f'<h1 style="color:#FF0000;font-size:18px;">{"Red = both ears heard the sound (3)"}</h1>', unsafe_allow_html=True)
  colorList = ['blue','orange','darkgreen','red']
  #we limit the data based on ear and id
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
  
  #plot of correct answers against the actual answer
  fig, ax = plt.subplots(1, 2,  sharex=False, sharey=False, figsize=(15,10))
  fig.suptitle('Correct answer'+"   vs   "+'Response', fontsize=20)
  ### count
  ax[0].title.set_text('count')
  order = dff.groupby(['Out','Response'])['Response'].count()
  order = order.reset_index(name = 'count')
  sns.barplot(data=order, x='Out', y='count', hue='Response', ax=ax[0])
  ax[0].grid(True)
  ### percentage
  ax[1].title.set_text('percentage')
  a = dff.groupby('Out')['Response'].count().reset_index()
  a = a.rename(columns={'Response':"tot"})
  b = dff.groupby(['Out','Response'])['Response'].count()
  b = b.reset_index(name = 'count')
  b = b.rename(columns={'count':0}).reset_index()
  b = b.merge(a, how="left")
  b["%"] = b[0] / b["tot"] *100
  sns.barplot(x='Out', y="%", hue='Response', data=b, ax=ax[1]).get_legend().remove()
  ax[1].grid(True)
  ### fix figure
  st.pyplot(fig)
  
  #Evalf correct responses
  index = dff.index
  responseEval = pd.DataFrame(columns = ['ear', 'answerTjek'])
  for i in range(len(dff)):
    if dff['Out'][index[i]] == dff['Response'][index[i]]:
      responseEval.loc[len(responseEval)] = [dff['Out'][index[i]], 1] #if answer is correct, then i'll set the value to 1
    else:
      responseEval.loc[len(responseEval)] = [dff['Out'][index[i]], 0] 
  
  answerListNoEar = ['Correct']
  noEarAnswer = responseEval[responseEval['ear'] == 0]['answerTjek']
  if noEarAnswer.value_counts().values[0] == noEarAnswer.count():
    if noEarAnswer[noEarAnswer.first_valid_index()] == 0:
      answerListNoEar = ['Not correct']
  else:
    answerListNoEar = ['Not Correct','Correct']
  
  answerListLeftEar = ['Correct']
  leftEarAnswer = responseEval[responseEval['ear'] == 1]['answerTjek']
  if leftEarAnswer.value_counts().values[0] == leftEarAnswer.count():
    if leftEarAnswer[leftEarAnswer.first_valid_index()] == 0:
      answerListLeftEar = ['Not correct']
  else:
    answerListLeftEar = ['Not Correct','Correct']
  
  answerListRightEar = ['Correct']
  rightEarAnswer = responseEval[responseEval['ear'] == 2]['answerTjek']
  if rightEarAnswer.value_counts().values[0] == rightEarAnswer.count():
    if rightEarAnswer[rightEarAnswer.first_valid_index()] == 0:
      answerListRightEar = ['Not correct']
  else:
    answerListRightEar = ['Not Correct','Correct']
  
  answerListBothEar = ['Correct']
  bothEarAnswer = responseEval[responseEval['ear'] == 3]['answerTjek']
  if bothEarAnswer.value_counts().values[0] == bothEarAnswer.count():
    if bothEarAnswer[bothEarAnswer.first_valid_index()] == 0:
      answerListBothEar = ['Not correct']
  else:
    answerListBothEar = ['Not Correct','Correct']
  
  st.write(f'<h1 style="color:#00000;font-size:22px;">{"Correctness of answers"}</h1>', unsafe_allow_html=True)
  fig, ax = plt.subplots(nrows=4, sharex=False, sharey=False, figsize = (10,14))
  ax[0].barh(answerListNoEar,noEarAnswer.value_counts().sort_index(), color = 'blue')
  totals = []
  for i in ax[0].patches:
    totals.append(i.get_width())
  total = sum(totals)
  for i in ax[0].patches:
    ax[0].text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12, color='black')
  ax[0].set_title('No ear', color = 'blue')
  
  ax[1].barh(answerListLeftEar,leftEarAnswer.value_counts().sort_index(),color = 'orange')
  totals = []
  for i in ax[1].patches:
    totals.append(i.get_width())
  total = sum(totals)
  for i in ax[1].patches:
    ax[1].text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12, color='black')
  ax[1].set_title('\n Left ear',color = 'orange')
  
  ax[2].barh(answerListRightEar,rightEarAnswer.value_counts().sort_index(),color = 'darkgreen')
  totals = []
  for i in ax[2].patches:
    totals.append(i.get_width())
  total = sum(totals)
  for i in ax[2].patches:
    ax[2].text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12, color='black')
  ax[2].set_title('\n Right ear',color = 'darkgreen')
  
  ax[3].barh(answerListBothEar,bothEarAnswer.value_counts().sort_index(),color = 'red')
  totals = []
  for i in ax[3].patches:
    totals.append(i.get_width())
  total = sum(totals)
  for i in ax[3].patches:
    ax[3].text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=12, color='black')
  ax[3].set_title('\n Both ears',color = 'red')
  st.pyplot(fig)
  
  #we make a list of unique frequencies and sort it
  listOfFreaquency = dff['Frekvens'].sort_values().unique()
  
  #we make a list of how many time the correct answer is given to a specific frequency
  listOfAccuracy = [1 if dff['Out'][index[i]] == dff['Response'][index[i]] else 0 for i in range(len(dff))]
  
  listOfFreaquencyAccuracy = pd.DataFrame(0,index=range(len(listOfFreaquency)),columns=list('AB'))
  for i in range(len(dff)):
    CurrentFreaquenzy = dff['Frekvens'][index[i]]
    indexOfFreaquenzy = np.where(listOfFreaquency == CurrentFreaquenzy)
    if listOfAccuracy[i] == 0:
      listOfFreaquencyAccuracy.loc[indexOfFreaquenzy[0][0],'A'] = listOfFreaquencyAccuracy['A'][indexOfFreaquenzy[0][0]]+1
    else:
      listOfFreaquencyAccuracy.loc[indexOfFreaquenzy[0][0],'B'] = listOfFreaquencyAccuracy['B'][indexOfFreaquenzy[0][0]]+1
  
  for i in range(len(listOfFreaquencyAccuracy)):
    count = listOfFreaquencyAccuracy['A'][i]+listOfFreaquencyAccuracy['B'][i]
    listOfFreaquencyAccuracy.loc[i,'A'] = listOfFreaquencyAccuracy['A'][i]/count*100
    listOfFreaquencyAccuracy.loc[i,'B'] = listOfFreaquencyAccuracy['B'][i]/count*100
  
  # st.write(listOfFreaquency)
  # st.write(listOfFreaquencyAccuracy)
  fig, ax = plt.subplots(ncols=1,nrows=2,figsize = (10,20))
  ax[0].set_title('Accuracy of freaquencies')
  
  for i in range(2):
    ax[i].plot(listOfFreaquency[round(len(listOfFreaquency)*i/2):round(len(listOfFreaquency)*(i+1)/2)],listOfFreaquencyAccuracy['B'][round(len(listOfFreaquency)*i/2):round(len(listOfFreaquency)*(i+1)/2)])
    ax[i].set_ylabel('Accuracy in %')
    ax[i].set_xlabel('Frequency')
  st.pyplot(fig)
