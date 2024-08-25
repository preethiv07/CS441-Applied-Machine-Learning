# Nearest Neighbors 

> Lecture - 2.2 Nearest Neighbors 
> Book- Section 1.2 
> **Classifying with Nearest Neighbors** 
> (p7-p10)

> ---

# Intro
- effective strategy
- We wish to predict the label y for any new example x; this is often known as a **query example or query.**
**CONCEPT:**
> if a labelled example xi is close to x, then p(y|x) is similar to p(y|xi).

**ISSUE:**
- if new example Xc is same distance to p(y = 1|x) and  p(y = −1|x)m they are in BOUNDARY.
- even small change can cause the data point to be in either of the class labels.

## Workaround = KNN
 - choose a label from k closet nearest neighbors 
 -  classifies new point with the class that has the **highest number of votes**
 -  K = 3 generally

## Practical Consideration of KNN
> 1. lot of labelled examples for the method to work
> 2. Need to **rescale**
    or **Whitening** (Transform the features so that the **covariance matrix **is the identity 
    (Issue: hard to do if the dimension is so large that the covariance matrix is hard to estimate.)
> 3.  difficulty is you need to be able to find the nearest neighbors for your query point
> It turns out that nearest neighbor in **high dimensions is one of those problems that is a lot harder than it seems**

**Note:**
usually enough to use an **approximate** nearest neighbor.

# Error Estimate for Nearest Neighbor
- cross-validation to estimate the error rate 
  1. Split the labelled training data into two pieces, a (typically large) training set and a (typically small) validation set. 
  2. take each element of the **validation** set and label it with the **label of the closest element** of the training set. 
  3. Compute the **fraction** of these attempts that produce an error (the true label and predicted labels differ). 
  4. Now **repeat** this for a different split, 
  5. **Average** the errors over splits. 

# Summary
1. Enough Training data
2. Good enough Low set of dimensions

---
# Fade out