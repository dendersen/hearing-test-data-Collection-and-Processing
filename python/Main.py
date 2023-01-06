import USB.USB_protocol as USB
import time 
import USB.integerSplit as inS
from Visualization.TreatedData import testSaveData

def main():
  pass

tone = 100

def runTest(ID, frequency, earPlayed):
  startTime = time.time()
  # For earPlayed: 0 = none, 1 = lefResponse, 2 = rightResponse, 3 = both
  correctedFrequency, delay = inS.generateFrequency(frequency,earPlayed)
  testAnswer = USB.sendMessage(delay)
  endTime = time.time()
  
  deltatime = round((endTime-startTime) * 1000,0)
  
  # secondsSpent = endTime.tm_sec - startTime.tm_sec
  # msSpent = endTime.tm_
  
  testSaveData(ID,correctedFrequency,earPlayed,testAnswer,deltatime)


runTest(1,5000,3)
# while(1):
#   USB.sendMessage(inS.generateFrequency(1000,1)[1])
#   time.sleep(5)
#   USB.sendMessage(inS.generateFrequency(500,0)[1])
#   time.sleep(5)






