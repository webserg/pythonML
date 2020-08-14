from collections import namedtuple

Pet = namedtuple("Pet", "type name age")
frank = Pet(type="pigeon", name="Френк", age=3)
print(frank.age)
print(frank)
frank = frank._replace(age=4)
print(frank)

class PetSlots:
    __slots__ = ("type", "name", "age")

    def __init__(self, type, name, age):
        self.type = type
        self.name = name
        self.age = age

frank_slots = PetSlots(type="pigeon", name="Френк", age=3)

# from dataclasses import dataclass
#
# @dataclass
# class PetData:
#     type: str
#     name: str
#     age: int
# frank_data = PetData(type="pigeon", name="Френк", age=3)
# print(frank_data)