import numpy as np
import scipy.misc as scm
import time


def main():
    k = 0.5
    pan = scm.imread('test_pan.jpg')
    mul =  scm.imread('test_mul.jpg')

    
    r = mul[:, :, 0]
    g = mul[:, :, 1]
    b = mul[:, :, 2]


    time_start = time.time()

    i = (r*0.171 +g*0.2+b*0.171)/0.632
    kx__pan_minus_iii = k*(pan-i)
    coe = pan/(i+kx__pan_minus_iii)
    nr = coe * (r+kx__pan_minus_iii)
    ng = coe * (g+kx__pan_minus_iii)
    nb = coe * (b+kx__pan_minus_iii)
    output_img = np.empty_like(mul)

    finish_time = time.time() - time_start
    print "finish time:", finish_time, " s"

    output_img[:, :, 0] = nr
    output_img[:, :, 1] = ng
    output_img[:, :, 2] = nb

    scm.imsave("output_by_run_p.jpg", output_img)

np.seterr(divide='ignore', invalid='ignore')
if __name__ == '__main__':
    main()
