from random import randint,random

class nodeNetwork:
  def __init__ (self,numberOfInputs:int, 
                NodeBiases:list[list[float]]
                ) -> None:
    """generates the nodes used for a neural network
    contains methods for running the network
    
    Args:
        numberOfInputs (int): the number of input nodes for the algorithm
        NodeBiases (list[list[float]]): the biases for processing nodes, also contains the number of layers and the number of nodes
    """
    
    self.inputNodes = [-1]*numberOfInputs
    self.processingNodes:list[list[Node]] = []
    
    for layerID,i in enumerate(NodeBiases):
      layer = [] 
      for nodeID,j in enumerate(i):
        if layerID == 0:
          node = Node(layerID,nodeID,numberOfInputs,self)
        else:
          node = Node(layerID,nodeID,len(NodeBiases[layerID-1]),self)
        node.setBias(j)
        layer.append(node)
      self.processingNodes.append(layer)
    
    print(self.inputNodes,self.processingNodes)
  
  def acces(self,layer:int,ID:int):
    if(layer == 0):
      return self.inputNodes[ID]
    return self.processingNodes[layer-1][ID].calculated

class Node:
  def __init__(self,nodeLayer:int,nodeID:int,connections:int,nodeNet:nodeNetwork) -> None:
    self.nodeLayer:int = nodeLayer
    self.nodeID:int = nodeID
    self.bias:float = 0
    self.weights:float = [1]*connections
    self.nodeNet:nodeNetwork = nodeNet
    self.calculated:float = 0
  
  def setBias(self,bias:float):
    self.bias = bias
  
  def randomizeBias(self,min:float, max:float,flip:bool):
    """randomly changes the bias of this node
    
    
    Args:
      min (float): the minimum amount of change/min of change
      max (float): the maximum amount of change/max of change
      flip (bool): describes if the previus parameters describe the range in which the change may happen or the amount of change. use true for amount of change
    
    if flip is true min and max will be considered as their absolute values
    """
    if flip:
      min = abs(min)
      max = abs(max)
      if randint(0,1) == 0:
        self.bias -= min+random()*(max-min)
      else:
        self.bias += min+random()*(max-min)
    else: # not flip
      self.bias += min+random()*(max-min)
    return
  
  def __str__(self) -> str:
    return (f"{{{self.nodeLayer},{self.nodeID},{self.bias}}}")
  
  def __repr__(self) -> str:
    return self.__str__()
  
  def randomizeWeight(self,min:float, max:float,flip:bool):
    """randomly changes the weihgts of ALL connections based on the given parameters
    
    
    Args:
      min (float): the minimum amount of change/min of change
      max (float): the maximum amount of change/max of change
      flip (bool): describes if the previus parameters describe the range in which the change may happen or the amount of change. use true for amount of change
    
    if flip is true min and max will be considered as their absolute values
    """
    if flip:
      min = abs(min)
      max = abs(max)
      for i in self.weights:
        if randint(0,1) == 0:
          i -= min+random()*(max-min)
        else:
          i += min+random()*(max-min)
    else: # not flip
      for i in self.weights:
        i += min+random()*(max-min)
    return
  
  def run(self):
    self.calculated = 0
    for i,w in enumerate(self.weights):
      self.calculated += self.nodeNet.acces(self.nodeLayer,i)*w
    pass

a = nodeNetwork(2,
                [
                [1,5,8],
                [2,5],
                [3,4,7]
                ]
)