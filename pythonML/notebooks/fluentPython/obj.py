class Gizmo:
    def __init__(self,k):
        print('Gizmo id: %d' % id(self))
    def go(self, y):
        print(k*y)

x = Gizmo(2)
print(x(2))