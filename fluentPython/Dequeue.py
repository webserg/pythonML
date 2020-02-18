class Dequeue:
    def __init__(self):
        self.items = []

    def add_head(self, e):
        self.items.insert(0, e)

    def add_tail(self, e):
        self.items.append(e)

    def remove_head(self):
        return self.items.pop(0)

    def remove_tail(self):
        return self.items.pop(0)


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