# Mutiple sample methods in Python

Here contains experiments demonstrating sample methods like ICDF, Metropolis-Hasting, slice sampling and etc

![fig](https://pico-bucket-test-1258276012.cos.ap-beijing.myqcloud.com/img/e5.png)
<p align="center" style="color:darkgray;font-size:10">
Cover image: Samples from a certain distribution via RMH method
</p>

## Dependency
Python version `>3.7.0` tested, following libraries are required:
```bash
pip install numpy, matplotlib, jupyter, ipykernel
```

## Demos
1. Sample methods

Interact with the notebook [`begin.ipynb`](https://github.com/PACRian/magorithm/blob/master/probalitity/begin.ipynb) which contains six experiments about different sampling methods
```bash
jupyter notebook begin.ipynb
```
Or, carry out another development via packages provided in the [`samplers.py`](https://github.com/PACRian/magorithm/blob/master/probalitity/samplers.py), for example:
```python
from samplers import gen_gibbs_sample
```
2. Map-array slice sampler

A slice sample methods implementation based on 2-dimensional gird in the file [`map_slice_sampler.py`](https://github.com/PACRian/magorithm/blob/master/probalitity/map_slice_sampler.py). Below is an example showing how `MapSlice` works
```python
# Generate example array
img = npr.random((10, 12))
plt.imshow(img)

# Mutiple choices to estabilish one slicer
# Check the `__init__` method documentation for more
slicer = MapSlice(img, ubound=[img.mean(), np.inf], do_subpixel=True)

# Do one sequential slice sampling
# Ignoring the proposal variables
samples = slicer.sample()[:, :-1]   

#TODO: Plot display
```


