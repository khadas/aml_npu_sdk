#!/usr/bin/env python3
import linecache
import os
import sys
def ExtractOutname(jsonpath=None,outpath="outnamelist.txt"):
	linenumber=0
	outline=[]
	if jsonpath is None:
		print('json path is none\n')
		return
	with open(jsonpath,"r") as fr:
		for line in fr:
			if line.find(r'"op": "output"') > 0:
				outline.append(linenumber)
			linenumber = linenumber+1
	with open(outpath,"w") as fw:
		for i, val in enumerate(outline):
			fw.write(linecache.getline(jsonpath,val).strip().replace(r'"name"',str(i))+os.linesep)
		fw.write("all output node number:"+str(len(outline)))
		
if __name__ == "__main__":
	file = sys.argv[1]
	ExtractOutname(file)