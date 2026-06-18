import re
import sys

for line in sys.stdin:
	if line[0:5] != "<edge":
		continue
	list = re.findall(r'"(.*?)"', line)
	output = ""
	for (i,e) in enumerate(list):
		if i == 0:
			output += (str)((int)(e[1:])+1) + " "
		else:
			output += e[1:] + " "
	print output
	



