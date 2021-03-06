# -*- coding: utf-8 -*-
import pyopencl as cl
import numpy as np
import scipy.misc as scm
import time

x = input("USE PYOPENCL?\n1.yes\n2.no\n")

pan = scm.imread('taipei_pan.jpg')
mul = scm.imread('taipei_mul.jpg')
output_img = np.empty_like(mul)

if x==2:
    k = 0.5
    np.seterr(divide='ignore', invalid='ignore')
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
    output_img[:, :, 0] = nr
    output_img[:, :, 1] = ng
    output_img[:, :, 2] = nb
elif x == 1:
    numrows = len(mul[:, 0, 0])
    numcols = len(mul[0, :, 0])
    pan_np = pan.ravel().astype(np.int8)
    r_np = mul[:, :, 0].ravel().astype(np.int8)
    g_np = mul[:, :, 1].ravel().astype(np.int8)
    b_np = mul[:, :, 2].ravel().astype(np.int8)
    res = np.empty_like(r_np)

    platform = cl.get_platforms()[1]   # Select the first platform [0]
    device = platform.get_devices()[0]  # Select the first device on this platform [0]
    #print device.name
    ctx = cl.Context([device])      # Create a context with your device
    #ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    time_start = time.time()
    pan_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pan_np)
    r_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=r_np)
    g_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=g_np)
    b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)
    res1_g = cl.Buffer(ctx, mf.WRITE_ONLY, res.nbytes)
    res2_g = cl.Buffer(ctx, mf.WRITE_ONLY, res.nbytes)
    res3_g = cl.Buffer(ctx, mf.WRITE_ONLY, res.nbytes)

    prg = cl.Program(ctx, """
    __kernel void run(__global const unsigned char *r_g, __global const unsigned char *g_g, __global const unsigned char *b_g,
     __global unsigned char *res1_g, __global unsigned char *res2_g, __global unsigned char *res3_g, __global const unsigned char *pan_g) {
      int gid = get_global_id(0);
      float i = (r_g[gid] * 0.171 + g_g[gid] * 0.2 + b_g[gid] * 0.171) / 0.632;
      float kx__pan_minus_iii = 0.5 * (pan_g[gid] - i);
      float coe = pan_g[gid] / (i + kx__pan_minus_iii);
      res1_g[gid] = coe * (r_g[gid] + kx__pan_minus_iii);
      res2_g[gid] = coe * (g_g[gid] + kx__pan_minus_iii);
      res3_g[gid] = coe * (b_g[gid] + kx__pan_minus_iii);
    }
    """).build()
    print "finish time:", time.time() - time_start,"s "

    time_start = time.time()
    prg.run(queue, pan_np.shape, (1024,), r_g, g_g, b_g, res1_g, res2_g, res3_g, pan_g)
    print "finish time:", time.time() - time_start,"s "

    time_start = time.time()
    cl.enqueue_copy(queue, res, res1_g)
    output_img[:, :, 0] = res.reshape((numrows,numcols))
    cl.enqueue_copy(queue, res, res2_g)
    output_img[:, :, 1] = res.reshape((numrows,numcols))
    cl.enqueue_copy(queue, res, res3_g)
    output_img[:, :, 2] = res.reshape((numrows,numcols))
    print "finish time:", time.time() - time_start,"s "
print "finish time:", time.time() - time_start,"s "
scm.imsave("output_by_run(array_v).jpg", output_img)