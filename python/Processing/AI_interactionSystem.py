class NodeArray:
  def __init__(self,numberOfLayer:int,nodesInLayer:list[int],bias:list[list[float]],multiplikation:list[list[list[float]]]) -> None:
    self.input = inputData()
    self.nodes:list[list[Node]] = [[]]
    for i in range(0,nodesInLayer[0]):
      self.nodes[0][i] = inputNode(i)
    
    for i,l in zip (range(1,numberOfLayer,nodesInLayer)):
      self.nodes[i][l] = Node([j for j in range(0,nodesInLayer[i-1])],self,i,bias[i,l],multiplikation[i,l])
    
  def acces(self,layer,index):
    return  self.nodes[layer,index]

class Node:
  def __init__(self,accesIndex:list[int],accesArray:NodeArray,layer:int,bias:float,multiplikation:list[float]) -> None:
    self.bias:list[float] = bias
    self.multiplikation:list[float] = multiplikation
    self.accesIndex:list[int] = accesIndex
    self.layer:int = layer
    self.nodeArray:NodeArray = accesArray
    self.values:list[float] = []*len(accesIndex)
  
  def collect(self):
    for i,j in enumerate(self.accesIndex):
      self.values[i] = self.nodeArray.acces(self.layer-1,j)
  
  def calculate(self):
    out = self.bias
    for i,l in enumerate(self.values):
      out += l*self.multiplikation[i]
    return

class inputNode(Node):
  def __init__(self, ownIndex) -> None:
    super().__init__(0, inputData, 0)

class inputData(NodeArray):
  def __init__(self,nodesInLayer) -> None:
    return