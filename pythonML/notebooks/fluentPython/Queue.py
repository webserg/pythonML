class Queue:
    def __init__(self):
        self.items = []

    def enqueue(self, e):
        self.items.insert(0, e)

    def dequeue(self):
        return self.items.pop()

    def __sizeof__(self):
        len(self.items)

    def is_empty(self):
        return len(self.items) == 0


if __name__ == '__main__':
    q = Queue()
    q.enqueue(1)
    q.enqueue(2)
    q.enqueue(3)
    q.enqueue(4)
    print(q.dequeue())
    print(q.dequeue())
    print(q.dequeue())
    print(q.dequeue())
    print(q.is_empty())
