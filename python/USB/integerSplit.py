def numberTo16Bit(num:int) -> str:
  out:str = ""
  out += (chr((num - num%255)//255))
  out += (chr(num%255))
  return out

def numberFrom16Bit(text:str) -> int:
  out:int = 0
  try:
    out += ord(text[1])
  except:
    pass
  out += ord(text[0])*255
  return out

numberFrom16Bit(numberTo16Bit(6666))
