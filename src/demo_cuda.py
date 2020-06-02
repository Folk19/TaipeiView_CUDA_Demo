# -*- coding: utf-8 -*-
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import scipy.misc as scm
import matplotlib.pyplot as p
import sys
import time
reload(sys)  
sys.setdefaultencoding('utf8')


def main():
	choose=input("Use PyCUDA or not?\n1. YES\n2. NO\n3. exit\n")
	a = np.random.randn(40000000).astype(np.float32)
	b = np.random.randn(40000000).astype(np.float32)
	dest = np.zeros_like(a)
	start_time = time.time()
	if choose == 1 :
		mod = SourceModule("""
		__global__ void add_them(float *dest, float *a, float *b)
		{
			const int i = threadIdx.x;
			dest[i] = a[i] + b[i];
		}
		""")
		add_them = mod.get_function("add_them")
		add_them(drv.Out(dest), drv.In(a), drv.In(b),block=(400,1,1))

	elif choose == 2:
		for i in range(40000000):
			dest[i]=a[i]+b[i]
	else:
		print "exit"
		sys.exit()

	if choose ==1 :
		print "\n-------Use GPU-------"
	elif choose ==2:
		print "\n-------Use CPU-------"
	else:
		pass
	print "Total time : ",time.time()-start_time," seconds"

if __name__ == '__main__':
	main()