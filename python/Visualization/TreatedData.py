import pandas as pd
from csv import DictWriter

def Change_Dataform():
  Re = pd.read_csv('Data\FinalResultStorage.csv')
  Svar = ["None","Left","Right","Both"]
  Ear = ["None","Left","Right","Both"]
  with open('NewResultStorage.csv', 'a') as f_object:
    for j in range(0,len(Re)):
      Out = Re.loc[j,'LeftOUT']+Re.loc[j,'RightOUT']*2
      Response = Re.loc[j,'LeftResponse']+Re.loc[j,'RightResponse']*2 # 0 = none, 1 = lefResponse, 2 = rightResponse, 3 = both
      data = {'ID': [Re.loc[j,'ID']],
              'Frekvens': [Re.loc[j,'Frekvens']],
              'Out': [Ear[Out]],
              'Response': [Svar[Response]],
              'AnswerTime': [Re.loc[j,'AnswerTime']]}
      df = pd.DataFrame(data)
      #append data frame to CSV file
      df.to_csv('NewResultStorage.csv', mode='a', index=False, header=False)

def testSaveData(ID, FrequencyPlayed, earPlayed, Answer, AnswerTime):
  with open('Data\FinalResultStorage.csv', 'a') as f_object:
    isEarLeftPlaying = 0
    isEarRightPlaying = 0
    
    if (earPlayed == 1 or earPlayed == 3):
      isEarLeftPlaying = 1
    if (earPlayed == 2 or earPlayed == 3):
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
  f = open("NewResultStorage.csv","w")
  f.write("ID,Frekvens,Out,Response,AnswerTime\n")
  f.close()