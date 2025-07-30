```python
import numpy as np
```

# Creating 1D , 2D , 3D , 4D , 5D array


```python
a1 = np.array([1,2,3,4,5,6])
a1
```




    array([1, 2, 3, 4, 5, 6])




```python
a2 = np.array([[2,4],
               [8,10]])
print(a2)
print(a2.shape)
```

    [[ 2  4]
     [ 8 10]]
    (2, 2)
    


```python
a3 = np.array([
    [[1,3],
     [2,4]],
    
    [[2,4],
     [7,9]]
])

print(a3)
print(a3.shape)
```

    [[[1 3]
      [2 4]]
    
     [[2 4]
      [7 9]]]
    (2, 2, 2)
    
🔑 Key Differences:

Feature	      |  a2	                     |        a3
Dimensionality| 2D	                     |       3D
Shape	      | (2, 2)	                 |       (1, 2, 2)
Axis count	  | 2 axes (rows, columns)	 |      3 axes (blocks, rows, columns)
Use case	  | Matrix operations	     |      Stacked matrices / tensor ops



```python
a4 =np.array([  
    [              #batch 1
        [[3,4,5],  #block 1.1
         [6,7,8]],
    
    [[1,1,1],      # block 1.2
     [2,2,2]]
    ],
    [               # batch 2
        [[9,10,11], # block 2.1
        [12,13,14]],
        
        [[3,3,3],   # block 2.2
        [4,4,4]]
    ]
])
a4
```




    array([[[[ 3,  4,  5],
             [ 6,  7,  8]],
    
            [[ 1,  1,  1],
             [ 2,  2,  2]]],
    
    
           [[[ 9, 10, 11],
             [12, 13, 14]],
    
            [[ 3,  3,  3],
             [ 4,  4,  4]]]])




```python
a4.shape
```




    (2, 2, 2, 3)




```python
a5 =np.array([
    [
        [              #batch 1
        [[3,4,5],  #block 1
         [6,7,8]],
    
    [[1,1,1],      # block 2
     [2,2,2]]
    ],
    [               # batch 2
        [[9,10,11], # block 1
        [12,13,14]],
        
        [[3,3,3],   # block 2
        [4,4,4]]
    ]
]
,
[
    [
    [[10,20,30],
    [40,50,60]],

    [[70,80,90],
    [100,110,120]]],

    [
    [[130,140,150],
    [160,170,180]],

     [[110,200,210],
     [220,230,240]]
    ]
]
])
a5
```




    array([[[[[  3,   4,   5],
              [  6,   7,   8]],
    
             [[  1,   1,   1],
              [  2,   2,   2]]],
    
    
            [[[  9,  10,  11],
              [ 12,  13,  14]],
    
             [[  3,   3,   3],
              [  4,   4,   4]]]],
    
    
    
           [[[[ 10,  20,  30],
              [ 40,  50,  60]],
    
             [[ 70,  80,  90],
              [100, 110, 120]]],
    
    
            [[[130, 140, 150],
              [160, 170, 180]],
    
             [[110, 200, 210],
              [220, 230, 240]]]]])




```python
a5.shape
```




    (2, 2, 2, 2, 3)



# Creating Array Using Different Methods



```python
arr = np.array([2,3,4],dtype=float)
arr
```




    array([2., 3., 4.])




```python
np.arange(2,13,4)      #start , stop , step
```




    array([ 2,  6, 10])




```python
np.arange(16).reshape(4,2,2)
```




    array([[[ 0,  1],
            [ 2,  3]],
    
           [[ 4,  5],
            [ 6,  7]],
    
           [[ 8,  9],
            [10, 11]],
    
           [[12, 13],
            [14, 15]]])




```python
np.arange(16).reshape(2,2,2,2)

```




    array([[[[ 0,  1],
             [ 2,  3]],
    
            [[ 4,  5],
             [ 6,  7]]],
    
    
           [[[ 8,  9],
             [10, 11]],
    
            [[12, 13],
             [14, 15]]]])




```python
np.ones((2,3))
```




    array([[1., 1., 1.],
           [1., 1., 1.]])




```python
np.zeros((3,3,3,3))
```




    array([[[[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]],
    
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]],
    
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]],
    
    
           [[[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]],
    
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]],
    
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]],
    
    
           [[[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]],
    
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]],
    
            [[0., 0., 0.],
             [0., 0., 0.],
             [0., 0., 0.]]]])




```python
np.zeros_like(a3)
```




    array([[[0, 0],
            [0, 0]],
    
           [[0, 0],
            [0, 0]]])




```python
np.ones_like(a5)
```




    array([[[[[1, 1, 1],
              [1, 1, 1]],
    
             [[1, 1, 1],
              [1, 1, 1]]],
    
    
            [[[1, 1, 1],
              [1, 1, 1]],
    
             [[1, 1, 1],
              [1, 1, 1]]]],
    
    
    
           [[[[1, 1, 1],
              [1, 1, 1]],
    
             [[1, 1, 1],
              [1, 1, 1]]],
    
    
            [[[1, 1, 1],
              [1, 1, 1]],
    
             [[1, 1, 1],
              [1, 1, 1]]]]])




```python
np.random.randint(2,10,size=[2,2,2],dtype='int64')
```




    array([[[6, 6],
            [9, 6]],
    
           [[8, 3],
            [6, 7]]])




```python
np.random.random((3,4))
```




    array([[0.94293141, 0.4805413 , 0.43671719, 0.24525736],
           [0.03590958, 0.91463973, 0.63899662, 0.38021616],
           [0.56677811, 0.12334269, 0.19945409, 0.13552213]])




```python
np.linspace(-10,10,100,dtype=int)
```




    array([-10, -10, -10, -10, -10,  -9,  -9,  -9,  -9,  -9,  -8,  -8,  -8,
            -8,  -8,  -7,  -7,  -7,  -7,  -7,  -6,  -6,  -6,  -6,  -6,  -5,
            -5,  -5,  -5,  -5,  -4,  -4,  -4,  -4,  -4,  -3,  -3,  -3,  -3,
            -3,  -2,  -2,  -2,  -2,  -2,  -1,  -1,  -1,  -1,  -1,   0,   0,
             0,   0,   0,   1,   1,   1,   1,   1,   2,   2,   2,   2,   2,
             3,   3,   3,   3,   3,   4,   4,   4,   4,   4,   5,   5,   5,
             5,   5,   6,   6,   6,   6,   6,   7,   7,   7,   7,   7,   8,
             8,   8,   8,   8,   9,   9,   9,   9,  10])




```python
np.linspace(-10,10,10,dtype=float)

```




    array([-10.        ,  -7.77777778,  -5.55555556,  -3.33333333,
            -1.11111111,   1.11111111,   3.33333333,   5.55555556,
             7.77777778,  10.        ])




```python
np.identity(6)
```




    array([[1., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 1.]])



# Array Attributes


```python
a1 = np.arange(10,dtype=np.int32)
a2 = np.arange(12,dtype=float).reshape(3,4)
a3 = np.arange(8).reshape(2,2,2)

print(a1)
print()
print(a2)
print()
print(a3)
```

    [0 1 2 3 4 5 6 7 8 9]
    
    [[ 0.  1.  2.  3.]
     [ 4.  5.  6.  7.]
     [ 8.  9. 10. 11.]]
    
    [[[0 1]
      [2 3]]
    
     [[4 5]
      [6 7]]]
    


```python
a1.ndim,a2.ndim,a3.ndim
```




    (1, 2, 3)




```python
a1.shape,a2.shape,a3.shape
```




    ((10,), (3, 4), (2, 2, 2))




```python
a1.size,a2.size,a3.size
```




    (10, 12, 8)




```python
a1.dtype,a2.dtype,a3.dtype
```




    (dtype('int32'), dtype('float64'), dtype('int64'))


np.itemsize in NumPy gives the size in bytes of each element (item) in a NumPy array.

```python
a1.itemsize,a2.itemsize,a3.itemsize
```




    (4, 8, 8)




```python
print(a1)
a1.astype(np.float32)
```

    [0 1 2 3 4 5 6 7 8 9]
    




    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=float32)




```python
arr1 = np.arange(12).reshape(3,4)
arr2 = np.arange(12,24).reshape(4,3)
arr1,arr2
```




    (array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]),
     array([[12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23]]))




```python
arr1 * 2
```




    array([[ 0,  2,  4,  6],
           [ 8, 10, 12, 14],
           [16, 18, 20, 22]])




```python
arr1 == 5
```




    array([[False, False, False, False],
           [False,  True, False, False],
           [False, False, False, False]])




```python
arr1 * arr2
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[32], line 1
    ----> 1 arr1 * arr2
    

    ValueError: operands could not be broadcast together with shapes (3,4) (4,3) 



```python
arr3 = np.arange(12,24).reshape(3,4)
```


```python
arr1 ** arr3
```


```python
arr4 = np.random.random((3,3))
arr4 = np.round(arr4*100)
arr4
```

# Array Functions


```python
max,min,sum,prod
```


```python
np.max(arr4)   # it gives from whole array
```


```python
np.max(arr4,axis=0)
```


```python
np.max(arr4,axis=1)
```


```python
np.min(arr4,axis=0)
```


```python
np.min(arr4,axis=1)
```


```python
np.sum(arr4)
```


```python
np.sum(arr4,axis=0)
```


```python
np.prod(arr4,axis=1)
```

### var, std,mean,median


```python
np.var(arr4,axis=1)
```


```python
np.std(arr4,axis=1)
```


```python
np.mean(arr4)
```


```python
np.median(arr4)
```

### sin , log , exp , round , floor , ciel , dot


```python
np.sin(arr4)
```


```python
np.exp(arr4)
```


```python
np.log(arr4)
```


```python
arr1 = np.arange(0,12).reshape(3,4)
arr2 = np.arange(24,36).reshape(4,3)
arr1,arr2
```


```python
np.dot(arr1,arr2)
```


```python
arr5 = np.random.random((3,5))
arr5
```


```python
arr5.round()
```


```python
np.round(arr5)
```


```python
np.floor(arr5)
```
+---------------------------------------------------------------+------------------------+
| Function     | Description                                | +3.7 | -3.7 | +2.5 | +3.5 |
+-------------+-------------------------------------+------+-------+------+------+-------+
| np.floor()  | Round down negative(−∞)                     | 3.0  | -4.0  | 2.0  | 3.0  |
| np.ceil()   | Round up positive(+∞)                       | 4.0  | -3.0  | 3.0  | 4.0  |
| np.trunc()  | Remove decimals (toward 0)                  | 3.0  | -3.0  | 2.0  | 3.0  |
| np.round()  | Round to nearest (half to even)             | 4.0  | -4.0  | 2.0  | 4.0  |
+-------------+-------------------------------------+------+-------+------+------+-------+

# Indexing and Slicing


```python
a1 = np.arange(10)
a2 = np.arange(12).reshape(3,4)
a3 = np.arange(8).reshape(2,2,2)

a3
```


```python
a1[0:3]
```


```python
a1[1::2]   # start,stop,step
```


```python
a2
```


```python
a2[1,1],a2[1,0]
```


```python
a3
```


```python
a3[0,1,1]
```


```python
a3[1,1,0]
```


```python
a2[1:3,1:3]      # row start:row stop , col start:col stop
```




    array([[ 5.,  6.],
           [ 9., 10.]])




```python
a2[0::2,0::2]    # row start:row stop:step , col start:col stop:step
```




    array([[ 0.,  2.],
           [ 8., 10.]])




```python
a2[1:,0:3:2]
```




    array([[ 4.,  6.],
           [ 8., 10.]])




```python
a2[0]
```




    array([0., 1., 2., 3.])




```python
a2[:,2]
```




    array([ 2.,  6., 10.])




```python
a2[1:3,1:3]
```




    array([[ 5.,  6.],
           [ 9., 10.]])




```python
a2
```




    array([[ 0.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]])




```python
a2[::2,::3]
```




    array([[ 0.,  3.],
           [ 8., 11.]])




```python
a2[::2,1::2]
```




    array([[ 1.,  3.],
           [ 9., 11.]])




```python
a2[1,::3]
```




    array([4., 7.])




```python
a2[:2,1:]
```




    array([[1., 2., 3.],
           [5., 6., 7.]])




```python
a3 = np.arange(27).reshape(3,3,3)
a3
```




    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8]],
    
           [[ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17]],
    
           [[18, 19, 20],
            [21, 22, 23],
            [24, 25, 26]]])




```python
a3[1]
```




    array([[ 9, 10, 11],
           [12, 13, 14],
           [15, 16, 17]])




```python
a3[::2]
```




    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8]],
    
           [[18, 19, 20],
            [21, 22, 23],
            [24, 25, 26]]])




```python
a3[0,1,:]
```




    array([3, 4, 5])




```python
a3[1,:,1]
```




    array([10, 13, 16])




```python
a3[2,1:,1:]
```




    array([[22, 23],
           [25, 26]])




```python
a3[0::2,0,0::2]
```




    array([[ 0,  2],
           [18, 20]])




```python
a3
```




    array([[[ 0,  1,  2],
            [ 3,  4,  5],
            [ 6,  7,  8]],
    
           [[ 9, 10, 11],
            [12, 13, 14],
            [15, 16, 17]],
    
           [[18, 19, 20],
            [21, 22, 23],
            [24, 25, 26]]])




```python

```

# Iterating


```python
for i in a1:
    print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    


```python
for i in a2:
    print(i)
    print()
```

    [0. 1. 2. 3.]
    
    [4. 5. 6. 7.]
    
    [ 8.  9. 10. 11.]
    
    


```python
for i in a3:
    print(i)
    print()
```

    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    
    [[ 9 10 11]
     [12 13 14]
     [15 16 17]]
    
    [[18 19 20]
     [21 22 23]
     [24 25 26]]
    
    

### np.nditer()


```python
for i in np.nditer(a3):
    print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    

# Reshaping


```python
a2.reshape(4,3)
```




    array([[ 0.,  1.,  2.],
           [ 3.,  4.,  5.],
           [ 6.,  7.,  8.],
           [ 9., 10., 11.]])



### Transpose


```python
a2.transpose()
```




    array([[ 0.,  4.,  8.],
           [ 1.,  5.,  9.],
           [ 2.,  6., 10.],
           [ 3.,  7., 11.]])




```python
a2.T
```




    array([[ 0.,  4.,  8.],
           [ 1.,  5.,  9.],
           [ 2.,  6., 10.],
           [ 3.,  7., 11.]])



### Ravel


```python
rav = a2.ravel()
```
when we update the value through ravel it will update the original matrix 

```python
rav[0] = 1
a2,rav
```




    (array([[ 1.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]]),
     array([ 1.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.]))




```python
flt = a2.flatten()
```


```python
flt[1] = 12
```


```python
a2,flt
```




    (array([[ 1.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]]),
     array([ 1., 12.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 10., 11.]))



# Stacking


```python
a2,arr2
```




    (array([[ 1.,  1.,  2.,  3.],
            [ 4.,  5.,  6.,  7.],
            [ 8.,  9., 10., 11.]]),
     array([[12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23]]))




```python

```
hstack rows remains same
vstack clos remains same

```python
np.hstack((a2,a2))
```




    array([[ 1.,  1.,  2.,  3.,  1.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.,  4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.,  8.,  9., 10., 11.]])




```python
np.vstack((a2,a2))
```




    array([[ 1.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.],
           [ 1.,  1.,  2.,  3.],
           [ 4.,  5.,  6.,  7.],
           [ 8.,  9., 10., 11.]])



# Splitting
hsplit split according to cols
vsplit split acc to rows

```python
a4
```




    array([[[[ 3,  4,  5],
             [ 6,  7,  8]],
    
            [[ 1,  1,  1],
             [ 2,  2,  2]]],
    
    
           [[[ 9, 10, 11],
             [12, 13, 14]],
    
            [[ 3,  3,  3],
             [ 4,  4,  4]]]])




```python
np.hsplit(a4,2)
```




    [array([[[[ 3,  4,  5],
              [ 6,  7,  8]]],
     
     
            [[[ 9, 10, 11],
              [12, 13, 14]]]]),
     array([[[[1, 1, 1],
              [2, 2, 2]]],
     
     
            [[[3, 3, 3],
              [4, 4, 4]]]])]




```python
np.vsplit(a4,2)
```




    [array([[[[3, 4, 5],
              [6, 7, 8]],
     
             [[1, 1, 1],
              [2, 2, 2]]]]),
     array([[[[ 9, 10, 11],
              [12, 13, 14]],
     
             [[ 3,  3,  3],
              [ 4,  4,  4]]]])]




```python
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8]])

```


```python
np.hsplit(a,2)
```




    [array([[1, 2],
            [5, 6]]),
     array([[3, 4],
            [7, 8]])]




```python
np.vsplit(a,2)
```




    [array([[1, 2, 3, 4]]), array([[5, 6, 7, 8]])]




```python

```


```python
 a11 = np.hstack((a,a))
a11
```




    array([[1, 2, 3, 4, 1, 2, 3, 4],
           [5, 6, 7, 8, 5, 6, 7, 8]])




```python
a12 = np.hsplit(a11,2)
print(a12)
type(a12)
```

    [array([[1, 2, 3, 4],
           [5, 6, 7, 8]]), array([[1, 2, 3, 4],
           [5, 6, 7, 8]])]
    




    list




```python
np.hsplit(a3,3)
```




    [array([[[ 0,  1,  2]],
     
            [[ 9, 10, 11]],
     
            [[18, 19, 20]]]),
     array([[[ 3,  4,  5]],
     
            [[12, 13, 14]],
     
            [[21, 22, 23]]]),
     array([[[ 6,  7,  8]],
     
            [[15, 16, 17]],
     
            [[24, 25, 26]]])]




```python
np.vsplit(a3,3)
```




    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]




```python
a3 = np.arange(12).reshape(3,4)
```


```python
a3
```




    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])




```python
np.hsplit(a3,2)
```




    [array([[0, 1],
            [4, 5],
            [8, 9]]),
     array([[ 2,  3],
            [ 6,  7],
            [10, 11]])]


