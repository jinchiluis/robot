from patchcore_exth import SimplePatchCore
import numpy as np  

model = SimplePatchCore()
model.load(path="patchcore_models/handy_512_30.pth")

model.debug_region_sizes("patch512training.PNG")