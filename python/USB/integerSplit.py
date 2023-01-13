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
  if(inputFrequency == 0):
    return (0,"010010")
  FrequencyMCU = adjustFrequencyToMCU(inputFrequency)
  
  us = round((1/FrequencyMCU*1000)/0.001/4)#us
  seconDigit = (us-(us%16))/16
  firstDigit = math.log(2**(us%16))/math.log(2)
  
  antalus = (firstDigit + seconDigit * 16)*4
  frequencySent = 10**6/antalus
  frequencyPlayed = adjustFrequencyFromMCU(frequencySent)
  
  #ændre delay til string
  second = str(int(seconDigit)) # 2
  first = str(int(firstDigit)) # 3
  #gør string pæn
  while len(second) < 3:
    second = "0" + second
  while len(first) < 2:
    first = "0" + first
  #ser om der er problemer
  if len(second) != 3 or len(first) != 2:
    raise Exception("length problems: first=" + str(first) +" second="+ str(second) + " frequency=" + str(inputFrequency))
  
  delayVar = first+second+str(earID)#gør delay til rigtig format
  
  return(frequencyPlayed,delayVar)

def adjustFrequencyToMCU(frequency:float):
  if(frequency  <= 4493.611475):
    return -11.71471053 + 1.081337202*frequency
  if(frequency <= 4729.103480):
    return min([400.7374078 + 1.046219379*frequency, -11.71471053 + 1.081337202*frequency])
  if(frequency <= 6475.952033):
    return 400.7374078 + 1.046219379*frequency
  if(frequency <= 6858.045559):
    return min([-370.0034962 + 1.226964164*frequency, 400.7374078 + 1.046219379*frequency])
  if(frequency <= 11672.20059):
    return -370.0034962 + 1.226964164*frequency
  if(frequency <= 13036.24341):
    return min([-5509.183026 + 1.810642549*frequency, -370.0034962 + 1.226964164*frequency])
  return -5509.183026 + 1.810642549*frequency

def adjustFrequencyFromMCU(frequency:float):
  if(frequency <= 5102.04081632653):
    return 0.924780908078687*frequency + 10.8335406437423
  if(frequency <= 7575.75757575758):
    return 0.955822478270657*frequency - 383.033822222363
  if(frequency <= 15625):
    return 0.815019728715651*frequency + 301.560149123402
  return 0.552290125081870*frequency + 3042.66738249958

# -11.71471053 + 1.081337202*frequency
# max([400.7374078 + 1.046219379*frequency, -11.71471053 + 1.081337202*frequency])
# 400.7374078 + 1.046219379*frequency
# max([-370.0034962 + 1.226964164*frequency, 400.7374078 + 1.046219379*frequency])
# -370.0034962 + 1.226964164*frequency
# max([-5509.183026 + 1.810642549*frequency, -370.0034962 + 1.226964164*frequency])
# -5509.183026 + 1.810642549*frequency

def generateFrequencyLegacy(inputFrequency:float,earID:int)->tuple[float,str]:
  
  
  #finder det nødvendige delay
  us = round((1/inputFrequency*1000)/0.001/4)#us
  seconDigit = (us-(us%16))/16
  firstDigit = math.log(2**(us%16))/math.log(2)
  
  #vi finder frekvensen som firstdigit og seconddigit beskriver
  antalus = (firstDigit + seconDigit * 16)*4
  frequencySent = 10**6/antalus
  #vi forudsiger hvilken frekvens der spilles
  if(frequencySent >= splitPoint):
    output = frequencySent* t_high_A+t_high_B
  else:
    output = frequencySent * t_low_A + t_low_B
  print('vi forventer tonen: ',output)
  
  #ændre delay til string
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