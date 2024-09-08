# SVM

# Agenda
![alt text](image-54.png)

- binary classifier

![alt text](image-56.png)

- more when actual and prediction is different
- zero when actual and prediction is same
- less near the boundary

# Limitation
![alt text](image-57.png)

# Overview
![alt text](image-58.png)
- same function positive  right of the boundary
- and left in the left qof the boundary
- Note "a" is vector with same size of "x"
- "b" ; scalar

# margin
![alt text](image-59.png)
- more robust 

When margin is less/narrow margin, its less robust to error
![alt text](image-60.png)

 > Cost
 ![alt text](image-61.png)

 - penalty is regularization to penalize the large error
  
# Goal
- reduce cost function
- minimize the error , so wider margin
  
# One feature
a,b are scalars
![alt text](image-62.png)
- This is straightforward
  
# Two features
- a is vector
- b is scalar

![alt text](image-63.png)

**Note**
- Y-intercept  = -b/a1
- slope(mx+b) = -A0/A1

![alt text](image-64.png)

# Generalization : K features
N - pairs
![alt text](image-65.png)

# Find Training error
Gamma(i) = predicted label
Yi = actual label

> *(gamma i is large, it is away from boundary (when sign of gamma is same as y, then it is a confident predict)*



# Hinge Loss
- if sign of actual and predict is different 
 ![alt text](image-69.png)

*Reference*
![alt text](image-68.png)

# Finding a and b
![alt text](image-70.png)

# Penalty Cost
![alt text](image-71.png)

- bigger the "A", larger the margin around "boundary"
- we need to keep it large "ENOUGH" just to classify the training examples to the correct side of the classification
- penalty - aTranspose * A /2.
- A Transpose X A = A squared

# Full Cost function
![alt text](image-72.png)
- we need to minimise it.
- We will use stochastic gradient to minimise it
  

# SVM Overview
  ![alt text](image-73.png)

Note:
- If digits curve feature is not sufficient for MNIST dataset, we can use pixel of image as a new feature
- We will see this in homework
- add feature to add is distance from origin to each of the features
- this allows features not separable to be separate
---
# Campuswire
![alt text](image-67.png)
When actual sign and predicted value sign is different, shouldnt this be actual X precit +1. Shouldnt it include output "y" as it is in the product?
why is it magnitude of gamma+1?

---