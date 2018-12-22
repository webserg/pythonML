import collections

Card = collections.namedtuple('Card', ['rank', 'suit'])


class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
                       for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]


beer_card = Card('7', 'diamonds')
deck = FrenchDeck()
print(len(deck))
print(deck[0])
print(deck[-1])

from random import choice
print(choice(deck))
print(deck[:3])
print(deck[12::13])
i=0
for card in deck:
    i+=1
    print(card)
print(i)

suit_values = dict(spades=3, hearts=2, diamonds=1, clubs=0)

def spades_high(card):
    rank_value = FrenchDeck.ranks.index(card.rank)
    return rank_value * len(suit_values) + suit_values[card.suit]

for card in sorted(deck, key=spades_high): # doctest: +ELLIPSIS
    print(card)

class Student:
    def __init__(self, name, grade, age):
        self.name = name
        self.grade = grade
        self.age = age
    def __repr__(self):
        return repr((self.name, self.grade, self.age))
    def weighted_grade(self):
        return 'CBA'.index(self.grade) / float(self.age)

student_objects = [
    Student('john', 'A', 15),
    Student('jane', 'B', 12),
    Student('dave', 'B', 10),
    ]

for s in sorted(student_objects, key=lambda student: student.age): # doctest: +ELLIPSIS
    print(s)

def myFunc(e):
    return len(e)

cars = ['Ford', 'Mitsubishi', 'BMW', 'VW']

cars.sort(reverse=True, key=myFunc)
print(cars)
cars.sort(reverse=True, key=lambda car: len(car) )
print(cars)
