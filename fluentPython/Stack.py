class Stack:
    def __init__(self):
        self.items = []

    def push(self, e):
        self.items.append(e)

    def pop(self):
        return self.items.pop()

    def peek(self):
        return self.items[len(self.items) - 1]

    def __sizeof__(self):
        return len(self.items)

    def is_empty(self):
        return self.items == []


if __name__ == '__main__':
    s = Stack()
    print(s.is_empty())
    s.push(4)
    s.push('dog')
    print(s.pop())
    print(s.pop())
