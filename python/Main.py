import random
import USB.USB_protocol as USB
import time 
import USB.integerSplit as inS
from Visualization.TreatedData import testSaveData
import math

class ToneObject:
  def __init__(self, tone, ear) -> None:
    self.tone = tone
    self.ear = ear

def runTest(ID, frequency, earPlayed):
  startTime = time.time()
  # For earPlayed: 0 = none, 1 = rightResponse, 2 = lefResponse, 3 = both
  correctedFrequency, delay = inS.generateFrequency(frequency,earPlayed)
  print(delay)
  testAnswer = USB.sendMessage(delay)
  endTime = time.time()
  
  deltatime = round((endTime-startTime) * 1000,0)
  
  testSaveData(ID,correctedFrequency,earPlayed,testAnswer,deltatime)

def runTestSequence(ID,minFrequency,maxFrequency,numberOfTones):
  a = math.exp(math.log((maxFrequency/minFrequency))/(numberOfTones-1))
  b = minFrequency
  
  listOfTones = []
  
  for x in range(numberOfTones):
    frequency = round(b*a**x)
    print(frequency)
    if(random.randint(1,100) <= 70):
      listOfTones.append(ToneObject(0,0))
    listOfTones.append(ToneObject(frequency,1))
    listOfTones.append(ToneObject(frequency,2))
    listOfTones.append(ToneObject(frequency,3))
  
  random.shuffle(listOfTones)
  
  i=0
  for tone in listOfTones:
    print('Der er '+str(len(listOfTones)-i)+' toner tilbage')
    i += 1
    runTest(ID,tone.tone,tone.ear)
    time.sleep(2)