import numpy as np

operations = {'add':'+','sub':'-','mul':'*','div':'/','pow':'^'}

class Derivative:
    @staticmethod
    def grad_operation_with_inp(op,const_ind):
        if(op.op=='add'):
            return 1
        elif(op.op=='sub'):
            if const_ind:
                return 1
            else:
                return -1
        elif(op.op=='mul'):
            if const_ind:
                return op.nB.value
            else:
                return op.nA.value
        elif(op.op=='div'):
            if const_ind:
                return 1/op.nB.value
            else:
                return -1*op.nA.value/(op.nB.value**2)
        elif(op.op=='pow'):
            if const_ind:
                return op.nB.value*(op.nA.value**(op.nB.value-1))
            else:
                return op.nA.value**op.nB.value*np.log(op.nA.value)
        elif(op.op=='log'):
            return np.log(op.nB.value)
    
    @staticmethod
    def grad_wrt_node(start_node,last_node):
        # if start_node is last_node then return 
        if(start_node==last_node):
            return 1
        
        # calculate grad_operation_with_inp(start_node.op,0) to treat nA as const
        # grad_operation_with_inp(start_node.op,1) to treat nB as const
        if(start_node.operation.nA.operation != None and start_node.operation.nB.operation != None):
            dnA = Derivative.grad_wrt_node(start_node.operation.nA,last_node)*Derivative.grad_operation_with_inp(start_node.operation,1)
            dnB = Derivative.grad_wrt_node(start_node.operation.nB,last_node)*Derivative.grad_operation_with_inp(start_node.operation,0)
        elif(start_node.operation.nA.operation == None or start_node.operation.nB.operation == None):
            if start_node.operation.const_ind():
                return Derivative.grad_wrt_node(start_node.operation.nA,last_node) * Derivative.grad_operation_with_inp(start_node.operation,1)
            else:
                return Derivative.grad_wrt_node(start_node.operation.nB,last_node) * Derivative.grad_operation_with_inp(start_node.operation,0)
        
        return dnA + dnB
        # grad_operation_with_inp(start_node.op,start_node.op.const_ind()) for const and var node
        
class Node:
    def __init__(self,constant=None,operation=None):
        # if constant==None and operation == None:
        #     raise Exception('Const and Operation not set')
        # elif constant!=None and operation!=None:
        #     raise Exception('Only one can be set for a node')
        self.const = constant
        self.operation = operation
        self.value = None
    
    def compute(self):
        if type(self.const) == np.ndarray or self.const!=None:
            self.value = self.const
            return
        else:
            self.value = self.operation.evaluate()
    
    def show(self):
        if self.operation != None:
            a = ''
            if(self.operation.nA.operation != None):
                a = self.operation.nA.show()
            else:
                a = self.operation.nA.const
            b = ''
            if(self.operation.nB.operation!=None):
                b = self.operation.nB.show()
            else:
                b = self.operation.nB.const
            return '('+str(a) + operations[self.operation.op] + str(b)+')'
        else:
            return self.const

    def __add__(self,other):
        return Operation(self,other,'add')
    
    def __sub__(self,other):
        return Operation(self,other,'sub')
    
    def __mul__(self,other):
        return Operation(self,other,'mul')
    
    def __pow__(self,other):
        return Operation(self,other,'pow')
    
    def __truediv__(self,other):
        return Operation(self,other,'div')


class Operation:
    '''
    operation -> add,sub,mul,div,pow,log(base e if nodeB is 0)
    '''
    def __init__(self,nodeA,nodeB,operation):
        self.nA = nodeA
        self.nB = nodeB
        self.op = operation
    
    def evaluate(self):
        self.nA.compute()
        self.nB.compute()
        
        if(self.op=='add'):
            return self.nA.value + self.nB.value
        elif(self.op=='sub'):
            return self.nA.value - self.nB.value
        elif(self.op=='mul'):
            return self.nA.value * self.nB.value
        elif(self.op=='div'):
            return self.nA.value / self.nB.value
        elif(self.op=='pow'):
            return self.nA.value ** self.nB.value
        elif(self.op=='log'):
            if self.nB == 0 :
                return np.log(self.nA.value)
            return np.log(self.nA.value) / np.log(self.nB.value)
    
    def const_ind(self):
        if self.nA.operation==None:
            return 0
        elif self.nB.operation==None:
            return 1
        return -1

a = Node(3)
#********
#   Single variable single variable node
#*******
# a1 = Node(operation=Node(-1)*a) #-z
# a2 = Node(operation=Node(np.e)**a1) #e^a1
# a3 = Node(operation=Node(1)+a2) # 1+a2
# a4 = Node(operation=Node(1)/a3) # 1/a3

# a4.compute()
# print(a4.value)
# print(a4.show())
# d = Derivative.grad_wrt_node(a4,a)
# print(d)

#********
#   Single variable double variable node
#*******

a1 = Node(operation=Node(2)*a)
a2 = Node(operation=Node(5)*a)
a3 = Node(operation=Node(operation=Node(np.e)**a1)+Node(operation=Node(np.e)**a2))
a4 = Node(operation=Node(np.e)**a2)
a5 = Node(operation=a3+a4)

a5.compute()
print(a5.value)
print(a5.show())
d = Derivative.grad_wrt_node(a5,a)
print(d)
