**Ignore missing**
![alt text](image-2.png)


### with nan

![alt text](image-1.png)

# Attempt 1
![alt text](image-3.png)

# Next attempt
![alt text](image-4.png)

# Attempt 2

![alt text](image-5.png)

**Note**
> Prevent numerical errors

In classification tasks, particularly with Naive Bayes classifiers, the computation of logs is typically done to prevent numerical errors. 
This is because multiplying many **small probabilities can lead to very small numbers, which can cause underflow issues.** 
By taking the logarithm of probabilities, the multiplication of probabilities becomes a sum of log-probabilities, which is numerically more stable and prevents such errors.