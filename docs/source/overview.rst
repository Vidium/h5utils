Overview
########

Pronounced "champy".
This library provides a set of helper tools for reading or writing to h5 files using the h5py library.

Description
***********

Ch5mpy provides a set of abstractions over h5py's (https://docs.h5py.org/en/stable/) objects for handling them as more commonly used objects :
- H5Dict: an object behaving as regular Python dictionaries, for exploring Files and Groups.
- H5List: an object behaving as regular Python lists for storing any set of objects.
- H5Array: an object behaving as Numpy ndarrays for dealing effortlessly with Datasets while keeping the memory usage low. This works by applying numpy functions to small chunks of the whole Dataset at a time.
- read/write utily functions for effortlessly storing any object to an h5 file.

Pickle
======

The first level of abstraction simply wraps h5py's Datasets, Groups and Files to allow pickling. Those objects can 
be directly imported from ch5mpy :

.. code:: python

    from ch5mpy import File
    from ch5mpy import Group
    from ch5mpy import Dataset


H5Dict
======

An H5Dict allows to explore the content of an H5 File or Group as if it was a regular Python dict. However, keys in 
an H5Dict are not loaded into memory until they are directly requested (unless they are small objects such as 0D 
Datasets). Large Datasets are wrapped as H5Arrays (see section [H5Arrays](#H5Arrays)).

To create an H5Dict, a `File` or `Group` object must be provided as argument :

.. code:: python

    from ch5mpy import File
    from ch5mpy import H5Dict
    from ch5mpy import H5Mode

    dct = H5Dict(File("backed_dict.h5", H5Mode.READ_WRITE))

    dct.keys()

    >>> H5Dict{a: 1, b: H5Array([1, 2, 3], shape=(3,), dtype=int64), c: {...}}


Here, `dct` is an H5Dict with 3 keys `a, b and c` where :
- `a` maps to the value `1` (a 0D Dataset)
- `b` maps to a 1D H5Array (values are not loaded into memory) 
- `c` maps to another H5Dict with keys and values not loaded yet

H5List
======

An H5List behave as regular Python lists, allowing to store and access any kind of object in an h5 file.
H5Lists are usually created when regular lists are stored in an h5 file. H5Lists can be created by providing that file, as for H5Dicts :

.. code:: python

    from ch5mpy import File
    from ch5mpy import H5List
    from ch5mpy import H5Mode

    lst = H5List(File("backed_list.h5", H5Mode.READ_WRITE))

    lst

    >>> H5List[1.0, 2, '4.']

.. code:: python

    class O_:
        def __init__(self, v: float):
            self._v = v

        def __repr__(self) -> str:
            return f"O({self._v})"

    lst.append(O(5.0))

    >>> H5List[1.0, 2, '4.', O(5.0)]


H5Lists can store regular integers, floats and strings, but can also store any object (such as the `O` object at index 3 in this example).

H5Array
=======

H5Arrays wrap Datasets and implement numpy arrays' interface to fully behave as numpy arrays while controlling the 
amount of RAM used. The maximum amount of available RAM for performing operations can be set with the class variable 
`H5Array.MAX_MEM_USAGE`, using suffixes `K`, `M` and `G` for expressing amounts in bytes.

H5Arrays can be created by passing a `Dataset` as argument. 

.. code:: python

    from ch5mpy import File
    from ch5mpy import H5Mode
    from ch5mpy import H5Array

    h5_array = H5Array(File("h5_s_array", H5Mode.READ_WRITE)["data"])


Then, all usual numpy indexing and methods can be used. 
When possible, those methods will be applied repeatedly on small chunks of the Dataset.

To load an H5Array into memory as a numpy array, simply run :

.. code:: python

    np.array(h5_array)
