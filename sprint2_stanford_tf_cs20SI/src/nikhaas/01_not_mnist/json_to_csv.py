# from http://stackoverflow.com/questions/1871524/how-can-i-convert-json-to-csv

import csv
import json
import sys
import flatdict

input = open(sys.argv[1])
data = json.load(input)
input.close()

flat_dict = flatdict.FlatDict(data[0])

output = csv.writer(sys.stdout)

output.writerow(flat_dict.keys())  # header row

for row in data:
    flat_dict = flatdict.FlatDict(row)
    output.writerow(flat_dict.values())
