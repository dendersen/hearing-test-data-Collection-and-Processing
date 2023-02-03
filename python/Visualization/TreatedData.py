import pandas as pd
from csv import DictWriter

def responseCheck(response:int, correct:int)->list[bool,bool]:
  """returns the correctnes on the left ear and right ear

  Args:
      response (int): what the user answeres
      correct (int): what the correct answer was

  Returns:
      list[bool,bool]: left ear, right ear
  """
  out:list[int]=[]
  out.append( 1 if (response % 2 == correct % 2) else 0) #left
  out.append( 1 if (response - response % 2 == correct - correct % 2)  else 0)#right
  return (out)
  

def Change_Dataform():
  finalResults = pd.read_csv('Data\FinalResultStorage.csv')
  idInformation = pd.read_csv('Data\ID_collection.csv')
  with open('Data\editResultStorage.csv', 'a') as a_object:
    for j in range(0,len(finalResults)):
      currentID = finalResults.loc[j,'ID']
      idInformationOfSpecificID = idInformation[idInformation['ID'] == currentID]
      Gender = idInformationOfSpecificID['Gender'].values
      Age:int = idInformationOfSpecificID['Age'].values
      HearingLoss = idInformationOfSpecificID['HearingLoss'].values
      HeadphoneTime = idInformationOfSpecificID['HeadphoneTime'].values
      # 0 = none, 1 = leftResponse, 2 = rightResponse, 3 = both
      Out = finalResults.loc[j,'LeftOUT']+finalResults.loc[j,'RightOUT']*2
      Response = finalResults.loc[j,'LeftResponse']+finalResults.loc[j,'RightResponse']*2 # 0 = none, 1 = leftResponse, 2 = rightResponse, 3 = both
      leftCorrect,rightCorrect = responseCheck(Response,Out)
      # we make two list that say if ear response was correct, 0 = false, 1 = true
      data = {'ID': [finalResults.loc[j,'ID']],
              'Frekvens': [finalResults.loc[j,'Frekvens']],
              'Out': [Out],
              'Response': [Response],
              'rightCorrect': [rightCorrect],
              'leftCorrect':[leftCorrect],
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
  f.write("ID,Frekvens,Out,Response,rightCorrect,leftCorrect,AnswerTime,Gender,Age,HearingLoss,HeadphoneTime\n")
  f.close()