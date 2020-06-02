import pyopencl as cl
plat = cl.get_platforms()
print plat[1].get_devices()
devices = plat[1].get_devices()
ctx = cl.Context([devices[0]])
print ctx.get_info(cl.context_info.DEVICES)