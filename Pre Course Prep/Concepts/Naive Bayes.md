# Intro
Naive Bayes is a simple yet powerful probabilistic machine learning algorithm 
- used for **classification** tasks.
- It's based on Bayes' Theorem and assumes that features in a dataset are **independent** of each other, which is why it's called "**naive**."

## Simplified Explanation with Example

**Scenario:**
Imagine you want to classify whether an email is spam or not spam (ham) based on the words it contains.

**1. Data Collection:**
You collect a bunch of emails and label them as "spam" or "ham."

**2. Feature Extraction:**
You break down each email into individual words (features).
**3. Calculate Probabilities:**
**Prior Probability: P(y)** The probability of an email being spam or ham regardless of the words it contains. For example:
> P(Spam) = (Number of Spam Emails) / (Total Number of Emails)
> P(Ham) = (Number of Ham Emails) / (Total Number of Emails)

**4. Likelihood: P(X|y)** The probability of each word given that an email is spam or ham. For example:
> P("buy" | Spam) = (Number of Spam Emails with the word "buy") / (Total Number of Spam Emails)
> P("buy" | Ham) = (Number of Ham Emails with the word "buy") / (Total Number of Ham Emails)

**Naive Assumption:** Assume that the presence of each word in the email is independent of the others. This simplifies the calculation.
**Calculate Posterior Probability:P(Y|x)** Use Bayes' Theorem to calculate the probability of an email being spam given the words it contains. 
For example, if an email contains the words "buy" and "cheap":
> P(Spam | "buy" and "cheap") = [P("buy" | Spam) * P("cheap" | Spam) * P(Spam)] / P("buy" and "cheap")

**6.Classify:** Compare the probabilities for "spam" and "ham" and classify the email based on which probability is higher.

# Example
Let's classify an email with the words "cheap" and "buy":

You have the following probabilities from your training data:
> - P(Spam) = 0.4 (40% of emails are spam)
> - P(Ham) = 0.6 (60% of emails are ham)
> - P("cheap" | Spam) = 0.5
> - P("cheap" | Ham) = 0.1
> - P("buy" | Spam) = 0.7
> - P("buy" | Ham) = 0.2


# How to calculate?
**Using Naive Bayes:**

1. Calculate the probability of the email being spam:
> P(Spam | "cheap" and "buy") ∝ P("cheap" | Spam) * P("buy" | Spam) * P(Spam)
= 0.5 * 0.7 * 0.4
= 0.14
2. Calculate the probability of the email being ham:
> P(Ham | "cheap" and "buy") ∝ P("cheap" | Ham) * P("buy" | Ham) * P(Ham)
= 0.1 * 0.2 * 0.6
= 0.012

***Since 0.14 > 0.012, the Naive Bayes classifier would classify this email as spam.***

# Key Points
**Simple:** Easy to implement and understand.
**Assumption:** Assumes features are independent, which might not always be true but works well in practice.
**Applications:** Commonly used for text classification (spam detection, sentiment analysis), medical diagnosis, and more.

---
# THE END