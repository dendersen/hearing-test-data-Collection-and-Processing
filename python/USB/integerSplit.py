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

def generateFrequency(inputFrequency:float)->tuple[float,str]:
  t_high = 1.73198537388486
  t_low =  1.21478044977032
  if(inputFrequency >= 17000):
    output = inputFrequency* t_high
  else:
    output = inputFrequency * t_low
  a = round(output)
  
  seconDigit = (a-(a%16))/16
  firstDigit = a%16
  
  second = str(seconDigit) # 2
  first = str(firstDigit) # 3
  
  while len(second) < 2:
    second = "0" + second
  while len(first) < 3:
    first = "0" + first
  
  if len(second) != 2 or len(first) != 3:
    raise Exception("length problems")
  print((output,first+second))
  return (output,first+second)

def delayFunctionality(delay:str):
  first = delay[0:3]
  second = delay[3:5]
  
  pass