def numberToBytes(num:int) -> bytes:
  bin:int = 2**8
  size:int = 1
  while True:
    if bin > num:
      out =  num.to_bytes(size,"big")
      return out
    
    bin = bin * 2**8
    size+=1

def numberFromBytes(input:bytes) -> int:
  out =  int.from_bytes(input,"big")
  return out

def byteSplit(input: bytes,*protocol:int) -> bytes:
  index:int = 0
  out:list[bytes] = []
  try:
    for i in protocol:
      out.append(input[index:index+i])
      index += i
  except (IndexError):
    print("an error has occured")
    print(f"the index: {index+i} is out of bounds for length: {len(input)}")
    pass
  return out

number:int = 10041247
print(numberToBytes(number))
print(numberFromBytes(numberToBytes(number)))
print(byteSplit(numberToBytes(number),2,1,3))
print([numberToBytes(number)[i:i+1] for i in range(0,3)])
