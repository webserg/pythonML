class Parent():
    def say(self, word):
        print(word)

    def dontsay(self, word):
        print(word)


class Child1(Parent):
    def say(self, word):
        print("child1" + word)


class Child2(Parent):
    def say(self, word):
        print("child2" + word)


def main():
    p = Parent()
    p.say("smth")
    c1 = Child1()
    c1.say(word = "smth")
    c2 = Child2()
    c2.say("smth")
    c2.dontsay("parent say")


if __name__ == "__main__":
    main()
