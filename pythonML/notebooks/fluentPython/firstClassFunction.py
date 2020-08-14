# Functions in Python are first-class objects. Programming language theorists define a
# “first-class object” as a program entity that can be:
#     • Created at runtime
# • Assigned to a variable or element in a data structure
# • Passed as an argument to a function
# • Returned as the result of a function


def factorial(n):
    '''returns n!'''
    return 1 if n < 2 else n * factorial(n - 1)

map(factorial, range(11))
