# Ch5mpy

Pronounced "champy".
Collection of helper tools for reading or writing to h5 files using the h5py library.

## Description

Ch5mpy provides a set of abstractions over h5py's (https://docs.h5py.org/en/stable/) objects for handling them as 
more commonly used objects.

### Pickle
The first level of abstraction simply wraps h5py's Datasets, Groups and Files to allow pickling. Those objects can 
be directly imported from ch5mpy :

```python
from ch5mpy import File
from ch5mpy import Group
from ch5mpy import Dataset
```

### H5Dict
An H5Dict allows to explore the content of an H5 File or Group as if it was a regular Python dict. However, keys in 
an H5Dict are not loaded into memory until they are directly requested (unless they are small objects such as 0D 
Datasets). Large Datasets are wrapped as H5Arrays (see section [H5Arrays](#H5Arrays)).

To create an H5Dict, a `File` or `Group` object must be provided as argument :

```python
from ch5mpy import File
from ch5mpy import H5Dict
from ch5mpy import H5Mode

dct = H5Dict(File("backed_dict.h5", H5Mode.READ_WRITE))

dct.keys()
```

```
H5Dict{a: 1, b: H5Array([1, 2, 3], shape=(3,), dtype=int64), c: {...}}
```

Here, `dct` is an H5Dict with 3 keys `a, b and c` where :
- `a` maps to the value `1` (a 0D Dataset)
- `b` maps to a 1D H5Array (values are not loaded into memory) 
- `c` maps to another H5Dict with keys and values not loaded yet

### H5Arrays
H5Arrays wrap Datasets and implement numpy arrays' interface to fully behave as numpy arrays while controlling the 
amount of RAM used. The maximum amount of available RAM for performing operations can be set with the class variable 
`H5Array.MAX_MEM_USAGE`, using suffixes `K`, `M` and `G` for expressing amounts in bytes.

H5Arrays can be created by passing a `Dataset` as argument. 

```python
from ch5mpy import File
from ch5mpy import H5Mode
from ch5mpy import H5Array

h5_array = H5Array(File("h5_s_array", H5Mode.READ_WRITE)["data"])
```

Then, all usual numpy indexing and methods can be used. 
When possible, those methods will be applied repeatedly on small chunks of the Dataset.

To load an H5Array into memory as a numpy array, simply run :

```python
np.array(h5_array)
```

### Roadmap

Logic functions
- [x] np.all
- [x] np.any
- [x] np.isfinite
- [x] np.isinf
- [x] np.isnan
- [ ] np.isnat
- [x] np.isneginf
- [x] np.isposinf
- [ ] np.iscomplex
- [ ] np.iscomplexobj
- [ ] np.isfortran
- [ ] np.isreal
- [ ] np.isrealobj
- [ ] np.isscalar
- [x] np.logical_and
- [x] np.logical_or
- [x] np.logical_not
- [x] np.logical_xor
- [ ] np.allclose
- [ ] np.isclose
- [x] np.array_equal
- [ ] np.array_equiv
- [x] np.greater
- [x] np.greater_equal
- [x] np.less
- [x] np.less_equal
- [x] np.equal
- [x] np.not_equal

Binary operations
- [ ] np.bitwize_and
- [ ] np.bitwize_or
- [ ] np.bitwize_xor
- [ ] np.invert
- [ ] np.left_shift
- [ ] np.right_shift
- [ ] np.packbits
- [ ] np.unpackbits
- [ ] np.binary_repr

String operations
- [ ] np.char.add
- [ ] np.char.multiply
- [ ] np.char.mod
- [ ] np.char.capitalize
- [ ] np.char.center
- [ ] np.char.decode
- [ ] np.char.encode
- [ ] np.char.expandtabs
- [ ] np.char.join
- [ ] np.char.ljust
- [ ] np.char.lower
- [ ] np.char.lstrip
- [ ] np.char.partition
- [ ] np.char.replace
- [ ] np.char.rjust
- [ ] np.char.rpartition
- [ ] np.char.rsplit
- [ ] np.char.rstrip
- [ ] np.char.split
- [ ] np.char.splitlines
- [ ] np.char.strip
- [ ] np.char.swapcase
- [ ] np.char.title
- [ ] np.char.translate
- [ ] np.char.upper
- [ ] np.char.zfill
- [ ] np.char.equal
- [ ] np.char.not_equal
- [ ] np.char.greater_equal
- [ ] np.char.less_equal
- [ ] np.char.greater
- [ ] np.char.less
- [ ] np.char.compare_chararrays
- [ ] np.char.count
- [ ] np.char.endswith
- [ ] np.char.find
- [ ] np.char.index
- [ ] np.char.isalpha
- [ ] np.char.isalnum
- [ ] np.char.isdecimal
- [ ] np.char.isdigit
- [ ] np.char.islower
- [ ] np.char.isnumeric
- [ ] np.char.isspace
- [ ] np.char.istitle
- [ ] np.char.isupper
- [ ] np.char.rfind
- [ ] np.char.rindex
- [ ] np.char.startswith
- [ ] np.char.str_len
- [ ] np.char.array
- [ ] np.char.asarray
- [ ] np.char.chararray

Mathematical functions
- [x] np.sin
- [x] np.cos
- [x] np.tan
- [x] np.arcsin
- [x] np.arccos
- [x] np.arctan
- [ ] np.hypot
- [ ] np.arctan2
- [ ] np.degrees
- [ ] np.radians
- [ ] np.unwrap
- [ ] np.deg2rad
- [ ] np.rad2deg
- [x] np.sinh
- [x] np.cosh
- [x] np.tanh
- [x] np.arcsinh
- [x] np.arccosh
- [x] np.arctanh
- [ ] np.around
- [ ] np.rint
- [ ] np.fix
- [x] np.floor
- [x] np.ceil
- [x] np.trunc
- [x] np.prod
- [x] np.sum
- [ ] np.nanprod
- [ ] np.nansum
- [ ] np.cumprod
- [ ] np.cumsum
- [ ] np.nancumprod
- [ ] np.nancumsum
- [ ] np.diff
- [ ] np.ediff1d
- [ ] np.gradient
- [ ] np.cross
- [ ] np.trapz
- [x] np.exp
- [x] np.expm1
- [x] np.exp2
- [x] np.log
- [x] np.log10
- [x] np.log2
- [x] np.log1p
- [ ] np.logaddexp
- [ ] np.logaddexp2
- [ ] np.i0
- [ ] np.sinc
- [ ] np.signbit
- [ ] np.copysign
- [ ] np.frexp
- [ ] np.ldexp
- [ ] np.nextafter
- [ ] np.spacing
- [ ] np.lcm
- [ ] np.gcd
- [x] np.add
- [ ] np.reciprocal
- [x] np.positive
- [x] np.negative
- [x] np.multiply
- [x] np.divide
- [x] np.power
- [x] np.subtract
- [x] np.true_divide
- [x] np.floor_divide
- [x] np.float_power
- [x] np.fmod
- [x] np.mod
- [ ] np.modf
- [ ] np.remainder
- [ ] np.divmod
- [ ] np.angle
- [ ] np.real
- [ ] np.imag
- [ ] np.conj
- [ ] np.conjugate
- [x] np.maximum
- [x] np.fmax
- [ ] np.amax
- [ ] np.nanmax
- [x] np.minimum
- [x] np.fmin
- [ ] np.amin
- [ ] np.nanmin
- [ ] np.convolve
- [ ] np.clip
- [x] np.sqrt
- [x] np.cbrt
- [x] np.square
- [x] np.absolute
- [x] np.fabs
- [x] np.sign
- [ ] np.heaviside
- [ ] np.nan_to_num
- [ ] np.real_if_close
- [ ] np.interp
