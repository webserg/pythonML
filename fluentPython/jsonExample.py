import json
from collections import namedtuple

json1_file = open('C:\\git\\highloadcup18-python\\data\\datatest.1.json')
json1_str = json1_file.read()
json1_data = json.loads(json1_str)
print(json1_data)
datapoints = json1_data['accounts']
print(datapoints)

x = json.loads(json1_str['accounts'], object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
print(x.Fname, x.sname, x.email)