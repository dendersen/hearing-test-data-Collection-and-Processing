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

t_low_A  =     0.828537603517999655
t_high_A =     0.564445942490515384
t_low_B  =     120.827362008712285
t_high_B =     2733.88736380325827

splitPoint = 14750

def generateFrequency(inputFrequency:float,earID:int)->tuple[float,str]:
  
  #går fra ønske til hvad skal spilles på mcu
  # if(inputFrequency >= splitPoint):
  #   output = inputFrequency* t_high_A+t_high_B
  # else:
  #   output = inputFrequency * t_low_A + t_low_B
  a = round((1/inputFrequency*1000)/0.001/4)
  
  #finder det nødvendige delay
  seconDigit = (a-(a%16))/16
  firstDigit = math.log(2**(a%16))/math.log(2)
  
  #vi finder frekvensen som firstdigit og seconddigit beskriver
  antalus = (firstDigit + seconDigit * 16)*4
  frequencySent = 10**6/antalus
  #vi forudsiger hvilken frekvens der spilles
  if(frequencySent >= splitPoint):
    output = frequencySent* t_high_A+t_high_B
  else:
    output = frequencySent * t_low_A + t_low_B
  print('vi forventer tonen: ',output)
  
  #øndre delay til string
  second = str(int(seconDigit)) # 2
  first = str(int(firstDigit)) # 3
  #gør string pæn
  while len(second) < 3:
    second = "0" + second
  while len(first) < 2:
    first = "0" + first
  #ser om der er problemer
  if len(second) != 3 or len(first) != 2:
    raise Exception("length problems")
  
  delayVar = first+second+str(earID)#gør delay til rigtig format
  output = delayFunctionality(delayVar)#finder ud af hvad der faktisk spilles
  
  #print("frequency = " + str(output),"\ndelay sring = " + str(delayVar))
  return (output,delayVar)

def delayFunctionality(delay:str) -> float:#finder ud af hvad der faktisk spilles
  M = int(delay[0:3]) 
  d = int(delay[3:5]) 
  
  c = d*16+M
  frequency = 250000/c
  
  output =(-frequency+t_low_B)/t_low_A 
  if output > splitPoint :
    output =(-frequency+t_high_B)/t_high_A 
  
  #print("unajusted:",frequency,"\nadjusted:",output)
  return output

generateFrequency(1000,3)[1]