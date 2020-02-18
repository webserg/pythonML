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
