def colorGenerator(numberOfColors:int):
  """_summary_

  Args:
      numberOfColors (int): minimum 8 colors
  """
  if (numberOfColors<8):
    numberOfColors = 8
  
  posibleCombinations:int = 0
  posibleCombinations+=255#red up
  posibleCombinations+=255#green up
  posibleCombinations+=255#red down
  posibleCombinations+=255#blue up
  posibleCombinations+=255#green down
  posibleCombinations+=255#red up
  posibleCombinations+=255#blue down
  
  colors = []
  rgbd:list[int] = [255,0,0,2]
  for i in range(0,numberOfColors):
    rgbd = split(rgbd[0],rgbd[1],rgbd[2],rgbd[3],posibleCombinations//numberOfColors)
    try:
      r:str=hex(rgbd[0]).replace("0x","")
      g:str=hex(rgbd[1]).replace("0x","")
      b:str=hex(rgbd[2]).replace("0x","")
      
      if(len(r) < 2):
        r="0"+r
      if(len(g) < 2):
        g="0"+g
      if(len(b) < 2):
        b="0"+b
      
      colors.append(("#" + r + g + b))
    except:
      print(rgbd[0],rgbd[1],rgbd[2])
  return colors


def split(r:int,g:int,b:int,direction:int,difference:int):
  if(direction == 0):
    r+=difference
    if(r>255):
      g+=r%255
      r=255
      direction+=1
    return [r,g,b,direction]
  if(direction == 1):
    b-=difference
    if(b<0):
      g+=abs(b)
      b=0
      direction+=1
    return [r,g,b,direction]
  if(direction == 2):
    g+=difference
    if(g>255):
      r-=g%255
      g=255
      direction+=1
    return [r,g,b,direction]
  if(direction == 3):
    r-=difference
    if(r<0):
      b+=abs(r)
      r=0
      direction+=1
    return [r,g,b,direction]
  if(direction == 4):
    b+=difference
    if(b>255):
      g-=b%255
      b=255
      direction+=1
    return [r,g,b,direction]
  if(direction == 5):
    g-=difference
    if(g<0):
      r+=abs(g)
      g=0
      direction=0
    return [r,g,b,direction]
  return [r,g,b,direction]

print(colorGenerator(10))