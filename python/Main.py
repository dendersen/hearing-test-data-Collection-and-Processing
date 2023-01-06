import USB.USB_protocol as USB
import time 
import USB.integerSplit as inS


def main():
  pass

tone = 100


while(1):
  USB.sendMesege(inS.generateFrequency(1000,3)[1])
  time.sleep(5)
  USB.sendMesege(inS.generateFrequency(500,3)[1])
  time.sleep(5)
  # USB.sendMesege(inS.generateFrequency(5000,3)[1])
  # time.sleep(5)
  # USB.sendMesege(inS.generateFrequency(4000,3)[1])
  # time.sleep(5)
  # USB.sendMesege(inS.generateFrequency(3000,3)[1])
  # time.sleep(5)





