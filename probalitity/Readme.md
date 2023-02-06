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

## Demo
To run the demos, interact with the notebook `begin.ipynb` which contains six experiments about different sampling methods
```bash
jupyter notebook begin.ipynb
```
Or, carry out another development via packages provided in the `samplers.py`, for example:
```python
from samplers import gen_gibbs_sample
```

