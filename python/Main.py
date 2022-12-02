import USB.USB_protocol as USB
import time 
import USB.integerSplit as inS


def main():
  pass

tone = 100

inS.generateFrequency(1000,3)

while(1):
  USB.sendMesege(inS.generateFrequency(1000,3)[1])
  time.sleep(2.4)



