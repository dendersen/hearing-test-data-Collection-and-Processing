def numberTo16Bit(num:int) -> str:
  out:str = ""
  out += (chr((num - num%512)//512))
  print(out)
  out += (chr(num%512))
  print(out)
  return out

def numberFrom16Bit(text:str) -> int:
  out:int = 0
  out += ord(text[1])
  print(out)
  out += ord(text[0])*512
  print(out)
  return out

numberFrom16Bit(numberTo16Bit(1234))
