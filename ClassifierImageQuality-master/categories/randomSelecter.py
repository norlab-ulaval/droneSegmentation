import math
import shutil, random, os
dirpath = 'images'

destDirectory = '../dataset'

classnames = os.listdir(os.getcwd())

for classname in classnames:
	if (os.path.isdir(os.getcwd() + "/" + classname)):
		allFileNames = os.listdir(classname)
		trainFileNames = random.sample(allFileNames, math.floor(len(allFileNames)*0.7))
		if not os.path.exists(destDirectory + "/train/" + classname):
	   		os.mkdir(destDirectory + "/train/" + classname)
		for fname in trainFileNames:
		    srcpath = os.path.join(classname + "/", fname)
		    shutil.move(srcpath, destDirectory + "/train/" + classname)

		allFileNames = os.listdir(classname)
		validFileNames = random.sample(allFileNames, math.floor(len(allFileNames) * 0.5))
		if not os.path.exists(destDirectory + "/valid/" + classname):
			os.mkdir(destDirectory + "/valid/" + classname)
		for fname in validFileNames:
			srcpath = os.path.join(classname + "/", fname)
			shutil.move(srcpath, destDirectory + "/valid/" + classname)

		testFileNames = os.listdir(classname)
		if not os.path.exists(destDirectory + "/test/" + classname):
			os.mkdir(destDirectory + "/test/" + classname)
		for fname in testFileNames:
			srcpath = os.path.join(classname + "/", fname)
			shutil.move(srcpath, destDirectory + "/test/" + classname)
