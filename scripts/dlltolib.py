import sys
import os
import ntpath

def main():
	filepath = sys.argv[1]

	if not os.path.isfile(filepath):
		print("File path {} does not exist. Exiting...".format(filepath))
		sys.exit()
	
	inname = ntpath.basename(filepath)
	deffile = open(inname+".def", "w")
	printlines = False
	start = False
	stop = False
	
	deffile.write("EXPORTS\n");
	
	with open(filepath) as fp:
		for line in fp:
			linewospace = line.strip()
			#deffile.write(linewospace[0:1]+'\n')
			if not printlines:
				printlines = linewospace.startswith("ordinal hint RVA")
			if not printlines:
				continue
			linearr = linewospace.split()
			if linearr and not start and linearr[0] == "1":
				start = True
				deffile.write(linearr[3]+'\n')
				continue
			if start and not stop:
				if not linearr:
					stop = True
					continue
				deffile.write(linearr[3]+'\n')
				
    
	deffile.close()
	print("Successfully wrote def file")

if __name__ == '__main__':
	main()
