class Node:
    def __init__(self,data):
        self.data=data
        self.next=None
a = Node(10)
b = Node(20)
c = Node(30)
a.next = b
b.next = c

current = a

while current:
    print(current.data)
    current = current.next