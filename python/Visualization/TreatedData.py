import pandas as pd
from csv import DictWriter

def Change_Dataform():
  finalResults = pd.read_csv('Data\FinalResultStorage.csv')
  idInformation = pd.read_csv('Data\ID_collection.csv')
  with open('Data\editResultStorage.csv', 'a') as a_object:
    for j in range(0,len(finalResults)):
      currentID = finalResults.loc[j,'ID']
      idInformationOfSpecificID = idInformation[idInformation['ID'] == currentID]
      Gender = idInformationOfSpecificID['Gender'].values
      Age = idInformationOfSpecificID['Age'].values
      HearingLoss = idInformationOfSpecificID['HearingLoss'].values
      HeadphoneTime = idInformationOfSpecificID['HeadphoneTime'].values
      
      Out = finalResults.loc[j,'LeftOUT']+finalResults.loc[j,'RightOUT']*2
      Response = finalResults.loc[j,'LeftResponse']+finalResults.loc[j,'RightResponse']*2 # 0 = none, 1 = rightResponse, 2 = lefResponse, 3 = both
      data = {'ID': [finalResults.loc[j,'ID']],
              'Frekvens': [finalResults.loc[j,'Frekvens']],
              'Out': [Out],
              'Response': [Response],
              'AnswerTime': [finalResults.loc[j,'AnswerTime']],
              'Gender': Gender,
              'Age': Age,
              'HearingLoss': HearingLoss,
              'HeadphoneTime': HeadphoneTime,
              }
      df = pd.DataFrame(data)
      #append data frame to CSV file
      df.to_csv('Data\editResultStorage.csv', mode='a', index=False, header=False)

def testSaveData(ID, FrequencyPlayed, earPlayed, Answer, AnswerTime):
  with open('Data\FinalResultStorage.csv', 'a') as f_object:
    isEarLeftPlaying = 0
    isEarRightPlaying = 0
    
    if (earPlayed == 2 or earPlayed == 3):
      isEarLeftPlaying = 1
    if (earPlayed == 1 or earPlayed == 3):
      isEarRightPlaying = 1
    
    Answer = [*Answer]
    
    DataToSave = {'ID': [ID],
                  'Frekvens': [FrequencyPlayed],
                  'LeftOUT': [isEarLeftPlaying],
                  'RightOUT': [isEarRightPlaying],
                  'LeftResponse': [int(Answer[0])],
                  'RightResponse': [int(Answer[1])],
                  'AnswerTime': [AnswerTime]}
    
    ListOfData = pd.DataFrame(DataToSave)
    # append data frame to CSV file
    ListOfData.to_csv('Data\FinalResultStorage.csv', mode='a', index=False, header=False)

def Clear_Data():
  f = open("Data\editResultStorage.csv","w")
  f.write("ID,Frekvens,Out,Response,AnswerTime,Gender,Age,HearingLoss,HeadphoneTime\n")
  f.close()
