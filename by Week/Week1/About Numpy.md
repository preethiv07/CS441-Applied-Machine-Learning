# Numpy

> - It is the ndarray object. 
> - This encapsulates n-dimensional arrays of **homogeneous** data types
> - with many operations being performed in compiled code for performance.
>
> ## Difference between Numpy and Python


| Numpy | Python |
|--------|--------|
| NumPy arrays have a fixed size at creation | Python lists (which can grow dynamically) |
| Changing the size of an ndarray will create a new array and delete the original. | - |
|  elements in a NumPy array are all same data type | Need not be same size |
| NumPy arrays facilitate advanced mathematical and other types of operations on large numbers of data. Typically, such operations are executed more efficiently  | - |
| scientific and mathematical Python-based packages are using NumPy arrays | - |

# Powerful features of Numpy
 1. vectorization 
 2.  broadcasting.

## Typical Python vs. Numpy

> Typical Python looping
```python
c = []
for i in range(len(a)):
    c.append(a[i]*b[i])
```

> same in C (more faster for millions of data in "a" and "b")
```
for (i = 0; i < rows; i++) {
  c[i] = a[i]*b[i];
}
```

> NUMPY (simple and as fast as "C")
```
c = a * b
```

# Why Numpy is fast?
## Vectorization
> Vectorization describes the absence of any explicit looping, indexing, etc., in the code 
> these things are taking place, of course, just “behind the scenes” i

-  more concise and easier to read
- easy to debug
- the code more closely resembles standard mathematical notation 
## Broadcasting 
> Broadcasting is the term used to describe the implicit element-by-element behavior of operations
> **Example:**
>> - in the example above, a and b could be multidimensional arrays of the same shape, or a scalar and an array, or even two arrays with different shapes, provided that the smaller array is “expandable” to the shape of the larger in such a way that the resulting broadcast is unambiguous