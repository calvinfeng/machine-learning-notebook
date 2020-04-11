# Naive Bayes Model

## Introduction

Using the Enron email data set, we will create a Naive Bayesian network in this simple exercise to
classify whether a given email is a spam or ham by looking at its word frequency as feature set.

## Mathematics

### Definitions

Let's define $N$ to be the number of total emails we have in the dataset and $N_{s}$ to be the
number of spam emails in the email set.

$N_{so}$ is the number of spam emails that contain the word "offer"

$N_{o}$ is the number of emails that contain the word "offer"

Then the probability of having a spam email in the set is said to be:

$$
P(SPAM=1) = \frac{N_{s}}{N}
$$

And the probability of having an email that contains the word *offer* is:

$$
P(OFFER=1) = \frac{N_{o}}{N}
$$

Finally, the conditional probability of an email being a spam email given that it contains the
word *offer*:

$$
P(SPAM=1\mid OFFER=1) := \frac{N_{so}}{N_{o}}
$$

### Postulate

If the probability of finding the word *offer* given that it's a spam email is higher than that of
finding the word *offer* in a non-spam email:

$$
P(OFFER =1 \mid SPAM=1)  > P(OFFER = 1 \mid SPAM=0)
$$

then we can infer that:

$$
P(SPAM=1 \mid OFFER=1) > P(SPAM = 1)
$$

### Proof

$$
P(SPAM=1 \mid OFFER=1) = \frac{P(OFFER=1 \mid SPAM=1)P(SPAM=1)}{P(OFFER=1)} = \frac{\frac{N_{so}}{N_{s}}\frac{N_{s}}{N}}{\frac{N_{o}}{N}} = \frac{N_{so}}{N_{o}}
$$

This is known as the **Bayes' rule**, famously stated as $P(A \mid B)=\frac{P(B \mid A)P(A)}{P(B)}$

$$
P(SPAM=0 \mid OFFER=1) = \frac{P(OFFER=1 \mid SPAM=0)P(SPAM=0)}{P(OFFER=1)} \\
P(SPAM=1 \mid OFFER=1) = \frac{P(OFFER=1 \mid SPAM=1)P(SPAM=1)}{P(OFFER=1)}
$$

For abbreviation, let's define that:

$$
P(SPAM=1) := P(S) \\
P(OFFER=1 \mid SPAM=1) := P(O \mid S) \\
P(OFFER=1 \mid SPAM=0) := P(O \mid S_{c}) \\
P(SPAM=1 \mid OFFER=1):= P(S \mid O)
$$

Begin with

$$
P(O \mid S) > P(O \mid S_{c})
$$

Rewrite them using **Bayes' rule**:

$$
\frac{P(S \mid O) P(O)}{P(S)} > \frac{P(S_{c} \mid O)P(O)}{P(S_{c})}
$$

The $P(O)$ terms cancel out each other:

$$
\frac{P(S \mid O)}{P(S)} > \frac{P(S_{c} \mid O)}{P(S_{c})}
$$

By definition, we can rewrite the right hand side as the following:

$$
\frac{P(S \mid O)}{P(S)} > \frac{1 - P(S \mid O)}{1 - P(S)}
$$

Re-organize the terms:

$$
\frac{1 - P(S)}{P(S)} > \frac{1 - P(S \mid O)}{P(S \mid O)}
$$

Then we can easily see that:

$$
\frac{1}{P(S)} - 1 > \frac{1}{P(S \mid O)} - 1 \\
\frac{1}{P(S)} > \frac{1}{P(S \mid O)} \\
$$

**Q.E.D.**
$$
P(S \mid O) > P(S)
$$

## Feature Probability

First of all, we load the data into a class object called `EmailSet` and compute the feature
probability for each word that has appeared in the email using `FeatureProbability.from_email_set`.

```python
from naive_bayes.email_set import EmailSet
from naive_bayes.email_set import build_and_save_email_set
from naive_bayes.feature_prob_set import FeatureProbabilitySet

# If you haven't pickled it, then run
build_and_save_email_set()

es = EmailSet.get()
fps = FeatureProbabilitySet.from_email_set(es)

print "Feature probability set has %s ham emails." % fps.class_count.ham_count
print "Feature probability set has %s spam emails." % fps.class_count.spam_count
```

    Dataset already processed!
    Feature probability set has 3672 ham emails.
    Feature probability set has 1500 spam emails.

```python
code = es.word_encoding_dictionary.word_to_code("offer")
print "Code:%s with count: %s" % (code, fps.code_count[code])
print "Prob ratio: %s" % fps.code_prob_ratio(code)
```

    Code:3751 with count: {'spam_count': 141, 'ham_count': 61}
    Prob ratio: 5.65849180328

### Edge cases

```python
code = es.word_encoding_dictionary.word_to_code("compensating")
print "Code:%s with count: %s" % (code, fps.code_count[code])
print "Prob ratio: %s" % fps.code_prob_ratio(code)
```

    Code:14526 with count: {'spam_count': 0, 'ham_count': 1}
    Prob ratio: 0.0

```python
code = es.word_encoding_dictionary.word_to_code("bacterial")
print "Code:%s with count: %s" % (code, fps.code_count[code])
print "Prob ratio: %s" % fps.code_prob_ratio(code)
```

    Code:20347 with count: {'spam_count': 1, 'ham_count': 0}
    Prob ratio: inf

Notice that the word *bacterial* and *compensating* have rare occurence in the data set. The
probability we compute has a very noisy estimate for their true value. In other words, they are
statistically insignificant for us to draw any reliable conclusion. It is not safe to make the
assumption that every email with teh word *bacterial* is a spam email.

### Filter Low-reach Features

Let's apply a limit to filter the words that have very low occurence in our data set.

```python
from naive_bayes.feature_prob_selector import FeatureProbabilitySelector
fps = FeatureProbabilitySet.from_email_set(es).filter_low_reach(limit=100)
best_spam_features = FeatureProbabilitySelector.best_spam_features(fps)
best_ham_features = FeatureProbabilitySelector.best_ham_features(fps)

print "Best Spam Features"
FeatureProbabilitySelector.print_feature_list(best_spam_features, es.word_encoding_dictionary)
print "\n"
print "Best Ham Features"
FeatureProbabilitySelector.print_feature_list(best_ham_features, es.word_encoding_dictionary)
```

    Best Spam Features
    18629 | 2004 | {'spam_count': 121, 'ham_count': 1} | 296.208
    2252 | microsoft | {'spam_count': 98, 'ham_count': 11} | 21.8094545455
    5912 | investment | {'spam_count': 96, 'ham_count': 11} | 21.3643636364
    2993 | results | {'spam_count': 98, 'ham_count': 18} | 13.328
    4144 | v | {'spam_count': 134, 'ham_count': 26} | 12.6166153846
    1123 | million | {'spam_count': 97, 'ham_count': 20} | 11.8728
    4335 | stop | {'spam_count': 147, 'ham_count': 31} | 11.6082580645
    6730 | software | {'spam_count': 101, 'ham_count': 22} | 11.2385454545
    2189 | 80 | {'spam_count': 104, 'ham_count': 23} | 11.0692173913
    515 | dollars | {'spam_count': 113, 'ham_count': 26} | 10.6393846154
    1035 | remove | {'spam_count': 110, 'ham_count': 28} | 9.61714285714
    7768 | stock | {'spam_count': 84, 'ham_count': 22} | 9.34690909091
    6072 | removed | {'spam_count': 83, 'ham_count': 22} | 9.23563636364
    674 | money | {'spam_count': 187, 'ham_count': 50} | 9.15552
    1089 | world | {'spam_count': 124, 'ham_count': 34} | 8.928
    3351 | save | {'spam_count': 125, 'ham_count': 35} | 8.74285714286
    201 | http | {'spam_count': 475, 'ham_count': 135} | 8.61333333333
    3868 | quality | {'spam_count': 101, 'ham_count': 29} | 8.52579310345
    5253 | canada | {'spam_count': 79, 'ham_count': 23} | 8.40834782609
    4643 | low | {'spam_count': 106, 'ham_count': 31} | 8.37058064516
    
    Best Ham Features
    3 | meter | {'spam_count': 0, 'ham_count': 773} | 0.0
    27 | cotten | {'spam_count': 0, 'ham_count': 157} | 0.0
    38 | aimee | {'spam_count': 0, 'ham_count': 121} | 0.0
    42 | daren | {'spam_count': 0, 'ham_count': 1030} | 0.0
    48 | fyi | {'spam_count': 0, 'ham_count': 277} | 0.0
    91 | mmbtu | {'spam_count': 0, 'ham_count': 527} | 0.0
    105 | hpl | {'spam_count': 0, 'ham_count': 1098} | 0.0
    113 | hplno | {'spam_count': 0, 'ham_count': 107} | 0.0
    115 | xls | {'spam_count': 0, 'ham_count': 504} | 0.0
    230 | sitara | {'spam_count': 0, 'ham_count': 405} | 0.0
    235 | pops | {'spam_count': 0, 'ham_count': 102} | 0.0
    2284 | scheduling | {'spam_count': 0, 'ham_count': 129} | 0.0
    242 | volumes | {'spam_count': 0, 'ham_count': 437} | 0.0
    325 | pat | {'spam_count': 0, 'ham_count': 249} | 0.0
    326 | clynes | {'spam_count': 0, 'ham_count': 184} | 0.0
    328 | enron | {'spam_count': 0, 'ham_count': 1462} | 0.0
    379 | nominations | {'spam_count': 0, 'ham_count': 133} | 0.0
    411 | hplc | {'spam_count': 0, 'ham_count': 124} | 0.0
    420 | hsc | {'spam_count': 0, 'ham_count': 134} | 0.0
    453 | 6353 | {'spam_count': 0, 'ham_count': 112} | 0.0

## Using All Features

### Unconditional Independence

Two variables are unconditionally independent, if knowing the result of one tells nothing of the
other under any circumstance.

For example, let `H` to be the event of flipping a head, and `S` to be the event of rolling a 6.

$$
P(S \wedge  H) = P(S)P(H) \\
P(H \mid S) = P(H) \\
P(S \wedge H) = P(S)P(H \mid S) = P(S)P(H)
$$

### Conditional Independence

Let's denote the event of having a particular disease to be $D$, event for showing positive on test
1 for detecting the disease to be $T_{1}$, and event for showing positive on test 2 for detecting
the same disease to be $T_{2}$.

The following case is **NOT** unconditionally independent because it is conditional

$$
P(T_{1} \mid T_{2}) \neq P(T_{1})
$$

Given that we know test 2 is showing a positive result, it does influence the probability of having
a positive on test 1, even though test 2 could have been a false positive. It is because a positive
result from either tests can influence the probability of having the disease. The tworesults we have
from $T_{1}$ and $T_{2}$ are connected by the variable $D$.

$$
P(T_{1} \mid D) \neq P(T_{1}) \\
P(T_{2} \mid D) \neq P(T_{2})
$$

Even though the two events are not unconditionally independent, under some cases, aka conditions,
they can be independent of each other.

$$
P(T_{1} \mid T_{2} \wedge D) = P(T_{1} \mid D)
$$

This is saying that if the condition of having the disease is satisfied, then the probabilities of
$T_{1}$ and $T_{2}$ are independent from each other. And equivalently speaking:

$$
P(T_{1} \wedge T_{2} \mid D) = P(T_{1} \mid D) \cdot P(T_{2} \mid D)
$$

### Examples When Unconditional Independence is Violated

Let's go back to the email examples. This is an example of conditional independence.

$$
P(LIMITED = 1 \mid OFFER = 1) \neq P(LIMIT = 1)
$$

The two random variables are not independent of each other in this particular condition because:

* The presence of the word *offer* suggests that the email is spam.
* If the email is spam, then it is more likely to contain the word *limited*.
* Therefore, the presence of the word *offer* makes the presence of the word *limited* more likely.
* In conclusion, the words *limited* and *offer* are **NOT** unconditionally independent. They are
  conditional independent in the sense that in some conditions they are dependent of each other and
  some conditions they are truly independent.

$$
P(LIMITED = 1 \mid OFFER = 1) > P(LIMITED = 1)
$$

However if we already knew the email is a spam email, then learning that we have the word *offer*
doesn't add any new knowledge to our estimate of probability of seeing the word *limited*.

$$
P(LIMITED = 1 \mid OFFER = 1 \wedge SPAM = 1) = P(LIMITED = 1 \mid SPAM = 1)
$$

### Examples When Conditional Independence is Violated

The words *limited* and *offer* often appear together in spam email because they frequently appear
as part of the compound phrase *a limited time offer*. In this sense, the probability of finding one
word while other one is present is definitely not independent.

$$
P(LIMITED=1 \mid SPAM=1 \wedge OFFER=1) > P(LIMITED=1 \mid SPAM=1)
$$

However, if we naively assume independence

$$
P(LIMITED=1 \wedge OFFER =1 \mid SPAM=1) = P(LIMITED=1 \mid SPAM=1) \cdot P(OFFER=1 \mid SPAM=1)
$$

It enables us to separate terms for ease of calculation.

## Applying The Naive Assumption

For abbreviation, I will write $LIMITED = 1$ as $L$ and $LIMITED = 0$ as $L_{c}$. Same applies to
other variables.

We are trying to calculate the probability ratio of the following:

$$
\frac{P(S \mid L \wedge O)}{P(S_{c} \mid L \wedge O)}
$$

Using Bayes' rule, we can rewrite them as:

$$
\frac{P(S)}{P(S_{c})} \cdot \frac{P(L \wedge O \mid S)}{P(L \wedge O \mid S_{c})}
$$

Using the **naive** assumption, we can break the terms apart and simplify the expression

$$
P(L \wedge O \mid S) = P(L \mid S) \cdot P(O \mid S) \\
P(L \wedge O \mid S_{c}) = P(L \mid S_{c}) \cdot P(O \mid S_{c})
$$

In conclusion,

$$
\frac{P(S)}{P(S_{c})} \cdot \frac{P(L \mid S)}{P(L \mid S_{c})} \cdot \frac{P(O \mid S)}{P(O \mid S_{c})}
$$

### More Generally Speaking

$$
\frac{P(S \mid W_{i} ... W_{N})}{P(S_{c} \mid W_{i} ... W_{N})} = \frac{P(S)}{P(S_{c})} \cdot \prod_{i}^{N} \frac{P(W_{i} \mid S)}{P(W_{i} \mid S_{c})}
$$

If we wish to improve the performance of the classifier, we should also include that probability
ratio of words that do not appear in the email. For example, in this Enron dataset, the word *enron*
NOT appearing in the email could be a good indicator that the email is spam.

If a word doesn't appear in the text/email, we just need to calculate

* *the probability of not having the word given an email is spam*
* *the probability of not having the word given an email is ham*.

$$
AbsenceRatio = \frac{P(W_{c} \mid S)}{P(W_{c} \mid S_{c})} = \frac{1 - P(W \mid S)}{1 - P(W \mid S_{c})}
$$

```python
from naive_bayes.naive_bayes_model import NaiveBayesModel
from naive_bayes.naive_bayes_model import classification_accuracy

model = NaiveBayesModel(fps)

ham_scores = model.email_scores(es.ham_emails)
spam_scores = model.email_scores(es.spam_emails)

cutoff_prob_ratios = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
for cutoff in cutoff_prob_ratios:
    result = classification_accuracy(cutoff, ham_scores, spam_scores)
    print result
```

    {'true_positive_rate': 0.998, 'false_positive_rate': 0.09994553376906318}
    {'true_positive_rate': 0.998, 'false_positive_rate': 0.09068627450980392}
    {'true_positive_rate': 0.998, 'false_positive_rate': 0.08088235294117647}
    {'true_positive_rate': 0.9966666666666667, 'false_positive_rate': 0.07053376906318083}
    {'true_positive_rate': 0.9966666666666667, 'false_positive_rate': 0.06018518518518518}
    {'true_positive_rate': 0.9913333333333333, 'false_positive_rate': 0.04847494553376906}
    {'true_positive_rate': 0.9753333333333334, 'false_positive_rate': 0.03567538126361656}
    {'true_positive_rate': 0.9493333333333334, 'false_positive_rate': 0.02423747276688453}

## Split into Training & Test Set

Let's split the data and see how well the model performs on unseen data!

```python
training_set, test_set = es.split(0.80)
print "Training set has %s ham emails and %s spam emails" % (len(training.ham_emails), len(training.spam_emails))
print "Test set has %s ham emails and %s spam emails" % (len(test.ham_emails), len(test.spam_emails))

fps = FeatureProbabilitySet.from_email_set(training_set).filter_low_reach(limit=100)
model = NaiveBayesModel(fps)

ham_scores = model.email_scores(test_set.ham_emails)
spam_scores = model.email_scores(test_set.spam_emails)

cutoff_prob_ratios = [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
for cutoff in cutoff_prob_ratios:
    result = classification_accuracy(cutoff, ham_scores, spam_scores)
    print result
```

    Training set has 3328 ham emails and 1344 spam emails
    Test set has 344 ham emails and 156 spam emails
    {'true_positive_rate': 0.9901639344262295, 'false_positive_rate': 0.08803301237964237}
    {'true_positive_rate': 0.9868852459016394, 'false_positive_rate': 0.08390646492434663}
    {'true_positive_rate': 0.9868852459016394, 'false_positive_rate': 0.07702888583218707}
    {'true_positive_rate': 0.9836065573770492, 'false_positive_rate': 0.06740027510316368}
    {'true_positive_rate': 0.9770491803278688, 'false_positive_rate': 0.0577716643741403}
    {'true_positive_rate': 0.9672131147540983, 'false_positive_rate': 0.048143053645116916}
    {'true_positive_rate': 0.9377049180327869, 'false_positive_rate': 0.037138927097661624}
    {'true_positive_rate': 0.898360655737705, 'false_positive_rate': 0.017881705639614855}
