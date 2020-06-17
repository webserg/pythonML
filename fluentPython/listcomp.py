symbols = '$¢£¥€¤'
codes = []
for symbol in symbols:
    codes.append(ord(symbol))
print(codes)

codes = [ord(symbol) for symbol in symbols]
print(codes)

items = []
items.append(1)
items.append(2)
items.append(3)
print(items.pop())

a_list = [1, '4', 9, 'a', 0, 4]

squared_ints = [e ** 2 for e in a_list if type(e) == int]

print(squared_ints)
# [ 1, 81, 0, 16 ]

print(list(map(lambda e: e ** 2, filter(lambda e: type(e) == int, a_list))))

# The list of lists
list_of_lists = [range(4), range(7)]
for x in list_of_lists:
    print('[')
    for y in x:
        print(y)
    print("]")
flattened_list = []

# flatten the lis
for x in list_of_lists:
    for y in x:
        flattened_list.append(y)

# The list of lists
list_of_lists = [range(4), range(7)]

# flatten the lists
flattened_list = [y for x in list_of_lists for y in x]

print(flattened_list)

# identity matrix

m = [[1 if row == col else 0 for col in range(0, 3)] for row in range(0, 3)]

print(m)

#Flatten a list of lists in one line
# The list of lists
list_of_lists = [range(4), range(7)]
flattened_list = []

# flatten the lis
for x in list_of_lists:
    for y in x:
        flattened_list.append(y)

# The list of lists
list_of_lists = [range(4), range(7)]

# flatten the lists
flattened_list = [y for x in list_of_lists for y in x]
