# Random Forest

# Agenda
- Classification with Decision Trees
![alt text](image-20.png)

# Intro
![alt text](image-21.png)
- Two features
- One is a binary feature that tells whether the image has curves or not, 
- the other indicates the number of vertices/corners in the image.
- Figure shows that we are using ONLY "has curves" to decide if image is 0 or 1.
 ![alt text](image-22.png)
 - There may by two trees that includes second feature "# of vertices"
 - This tree is deeper
  
  # Decision Tree
  ![alt text](image-23.png)
  - Label of the leaf node gives the "class" for a given input
  
  # Classification as If then else rules
  ![alt text](image-25.png)

  # Construction
  - It is recursive
  - We stop when
  - 1. depth > max depth (dont want tree to be too deep)
  - 2. size data < min leaf size (we dont want overfitting)
  - 3. All elements are in same class

  ![alt text](image-27.png)
- Subset l and subset r = best split determined by algorithm (statistic test)
- children node is created based on split and each branch will have only subset that passed the test from the parent node.
- Once algorithm moved forward, it doesnt back track to re-consider the earlier choices

# Best Split
dataset = S
feature = Xi

![alt text](image-28.png)

- based on feature. find subset that has more information gain
- Information gain is higher when classes have memebers from one class and less when it is diverse
(concept is Entropy)
![alt text](image-29.png)
- bottom has higher Information gain
- we compute information gain with respect to subsets and comparing it to the original set.

# Which Feature and What is the feature value?
## Cannot be sorted
![alt text](image-31.png)

- choose m = number of featues
- m  < total number of featues
- We select the feature and based on if the members can be sorted or not, we follow two approaches
- If small number of feature, check if it is within the threshold and if value matches the one subset, assign it here..remaining go to the second subset
- If Large, using normal distribution, assign half of the probable values to one subset and remaining goes to the other
- Example: weather, raining, sunny, snow

## Features that can be SORTED
![alt text](image-34.png)
- there may be many thresholds.
- In example above on top, threshold is at 2.5. 
- It can be at 1.5
- boundary is the threshold
- set of "k" examples, there is k-1 splits
- we choose them at random. common choice for random forest.

# Learning
- target is dataset assigned to class

![alt text](image-35.png)

# Limitation
![alt text](image-36.png)
- overfitting is training samples isnt large enough

# Random Foreset
![alt text](image-37.png)

# Building Random Forests- simple strategy
![alt text](image-38.png)

## Bagging
![alt text](image-39.png)

---
# Fade Out