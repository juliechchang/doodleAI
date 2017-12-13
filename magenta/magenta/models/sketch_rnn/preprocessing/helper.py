from PIL import Image
import numpy as np

def view_npy(infile,outfile):
	data = np.load(infile)
	img = Image.fromarray(data)
	img.save(outfile)

def truncate_npz(limit,outfile):
	#convert npz file
	x = np.load('cat.npz')
	np.savez(outfile,train= x['train'][:limit],valid=x['valid'],test=x['test'])
    

