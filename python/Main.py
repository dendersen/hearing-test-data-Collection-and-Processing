import USB.USB_protocol as USB
import time 
import USB.integerSplit as inS
from Visualization.TreatedData import testSaveData

def main():
  pass

tone = 100

def runTest(frequency, earPlayed):
  
  # For earPlayed: 0 = none, 1 = lefResponse, 2 = rightResponse, 3 = both
  testAnswer = USB.sendMessage(inS.generateFrequency(frequency,earPlayed)[1])
  
  testSaveData()

while(1):
  USB.sendMessage(inS.generateFrequency(1000,1)[1])
  time.sleep(5)
  USB.sendMessage(inS.generateFrequency(500,0)[1])
  time.sleep(5)






