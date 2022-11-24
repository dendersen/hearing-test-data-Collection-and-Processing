def numberTo16Bit(num:int) -> str:
  out:str = []
  out += (chr((num - num%256)//256))
  out += (chr(num%256))
  return out

def numberFrom16Bit(text:str) -> int:
  out:int = 0
  out += ord(text[0])
  out += ord(text[1])
  return out

