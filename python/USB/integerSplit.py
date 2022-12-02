import math

def numberTo16Bit(num:int) -> str:
  out:str = ""
  out += (chr((num - num%512)//512))
  out += (chr(num%512))
  return out

def numberFrom16Bit(text:str) -> int:
  out:int = 0
  out += ord(text[1])
  out += ord(text[0])*512
  return out

numberFrom16Bit(numberTo16Bit(1234))

def generateFrequency(inputFrequency:float,earID:int)->tuple[float,str]:
  t_high = 1#.73198537388486
  t_low =  1#.21478044977032
  if(inputFrequency >= 17000):
    output = inputFrequency* t_high
  else:
    output = inputFrequency * t_low
  a = round((1/output*1000)/0.001/4)
  
  seconDigit = (a-(a%16))/16
  firstDigit = a%16
  
  second = str(int(seconDigit)) # 2
  first = str(int(firstDigit)) # 3
  
  while len(second) < 3:
    second = "0" + second
  while len(first) < 2:
    first = "0" + first
  
  if len(second) != 3 or len(first) != 2:
    raise Exception("length problems")
  
  delayVar = first+second+str(earID)
  output = delayFunctionality(delayVar)
  
  print((output,delayVar))
  return (output,delayVar)

def delayFunctionality(delay:str) -> float:
  first = int(delay[0:3])
  second = int(delay[3:5])
  delay_us = first+second*16 #ns
  
  frequecy = 1000 / (delay_us * 4 * 0.001)
  
  print(frequecy)
  return frequecy

generateFrequency(1000,3)[1]