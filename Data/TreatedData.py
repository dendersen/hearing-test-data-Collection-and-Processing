import pandas as pd
from csv import DictWriter

Re = pd.read_csv('FinalResultStorage.csv')
field_names = ['ID','Frekvens','Out','Response','AnswerTime']
Svar = ["None","Left","Right","Both"]
Ear = ["None","Left","Right","Both"]

def Change_Dataform():
  # with open('NewResultStorage.csv', 'a') as f_object:
  for j in range(0,len(Re)):
    Out = Re.loc[j,'LeftOUT']+Re.loc[j,'RightOUT']*2
    Response = Re.loc[j,'LeftResponse']+Re.loc[j,'RightResponse']*2
    data = {'ID': [Re.loc[j,'ID']],
            'Frekvens': [Re.loc[j,'Frekvens']],
            'Out': [Ear[Out]],
            'Response': [Svar[Response]],
            'AnswerTime': [Re.loc[j,'AnswerTime']]}
    df = pd.DataFrame(data)
    #append data frame to CSV file
    df.to_csv('NewResultStorage.csv', mode='a', index=False, header=False)

def Clear_Data():
  f = open("NewResultStorage.csv","w")
  f.write("ID,Frekvens,Out,Response,AnswerTime\n")
  f.close()
