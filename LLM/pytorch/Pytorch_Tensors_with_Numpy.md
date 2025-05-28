# Pytorch Tensors with Numpy



* Numpy Interoperability

  

  ```python
  import torch
  import numpy as np
  
  points = torch.ones(3, 5)
  point_np = points.numpy()
  ```

  

* Serializing tensors

  ```python
  torch.save(points, './data/ourpoints.t')
  
  or
  
  with open('./data/ourpoints.t', 'wb') as f:
  	torch.save(points, f)
  ```



* Load the Serialized tensors

```python
points = torch.load('./data/ourpoints.t')

or

with open('./data/ourpoints.t', 'rb') as f:
	points = torch.load(f)
```



* Serializing to HDF5 with h5py

  * Save tensors with interoperablity
  * HDF5 format (www.hdfgroup.org/solutions/hdf5)
  * HDF5 is a portable, widely supported format for representing serialized multidimensional arrays

  ```python
  !pip install h5py
  import h5py
  
  # Save tensors in hdf5 format
  f = h5py.File('./data/ourpoints.hdf5', 'w')
  dset = f.create_dataset('coords', data=points.numpy())
  f.close()
  
  # Load tensors from hdf5 file
  f = h5py.File('./data/ourpoints.hdf5', 'r')
  dset = f['coords']
  last_points = dset[-2:]
  last_points = torch.from_numpy(dset[-2:])
  f.close()
  ```

  

