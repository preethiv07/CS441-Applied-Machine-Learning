# Agenda:
![alt text](image-40.png)

# Decision tree
![alt text](image-41.png)

# Construction
-recursive alogorithm
![alt text](image-42.png)

# Entropy
- data set S  can be in one of two classes A or B
- A : could be set of 1s
- B: could be st of 0s
- N: sum of elements in A and B
- Proportion of A and B (same of poabblity of A and B)

![alt text](image-43.png)

# Entropy
![alt text](image-44.png)

- when same number of elements in A and B, Entropy is max(1)
- As and when more elements are in one class than other, entropy reduces


**Sample**
![alt text](image-45.png)
- 100 elements
- Even when A switch to B, the entropy remains the same
**MORE..**
![alt text](image-46.png)

![alt text](image-47.png)
- note Proportion of B = 1 - P(A)
- base of log is "2". but, depends on application

# General case with more class
-  "C" different possible Classes
  ![alt text](image-48.png)

> More Entropy, More diverse set
> Goal is to reduce Entropy

  # Information gain
  - we want the Entropy of the split set to be less diverse than Entropy of the original set
  - "Information gain" - Informational Theory
![alt text](image-49.png)

- Note , The proportion for right and left subset provides weightage of that subset which is used to multiple with the Entropy of the respective subset to calcualte the overall Entropy (After split)
..Generalising to "n" subsets
![alt text](image-50.png)

# Comparative study
![alt text](image-52.png)

# Dealing with Missing value
![alt text](image-53.png)

---
### Fade Out..