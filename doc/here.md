# How does it work

### Basics

Notice that we assume all values follow a normal / gaussian distribution.
This is not necessarily true, but it is "precise enough" for the classifier to get 97% (or more) precision.

The classifier itself is, in fact, really simple. <br>
It's called **Naive Bayesian classifier** is based on the bayes formula:

$$ P(A|B) = \frac{P(B|A) P(A)}{P(B)} $$

where $P(A|B)$ is the probability of the event A given that event B holds.

In this specific dataset we want to calculate:

$$ P(Adelie|x_1,x_2, \dots \ , x_n) = \frac{P(x_1,x_2, \dots \ , x_n|Adelie) P(Adelie)}{P(x_1,x_2, \dots \ , x_n)} $$

where $x_1,x_2, \dots , x_n$ are some data of a penguin (bill lenght, flipper lenght, ...), for each specie (Adelie, Chinstrap, Gentoo), compare them and choose the specie with the higher probability.

Notice that $P(x_1,x_2, \dots \ , x_n)$ doesn't really need to be evaluated, since dividing two number for the same positive value does not change their order. So we can just ignore it and get the even faster:

$$P(x_1,x_2, \dots \ , x_n|Adelie) P(Adelie)$$

Let's look to the specific components of the formula:
* $P(Adelie)$ is really easy to get, since it is the same as the frequency of the specie divided by the number of total samples.
* $P(x_1,x_2, \dots \ , x_n|Adelie)$ this is the thoughest and needs a chapter for itself.


### How do we calculate that

$P(x_1,x_2, \dots , x_n|Adelie)$ is a distribution on n variables. Notice that there are ways to calculate this type of distribution and the classifiers who do that are called **Optimal Bayesian classifiers**. They reach the best accuracy as possible ([literally](https://en.wikipedia.org/wiki/Bayes_classifier)) but can be incredibly expensive from the computational point of view.

This is why we often assume that the single variables are independent between them. This is not necessarily true and can lead to errors, but at the same time it's ridicoluosly good to make computation faster. We can then rewrite it as:

$$P(x_1,x_2, \dots \ , x_n|Adelie)=P(x_1|Adelie)\cdot P(x_2|Adelie)\cdot \dots \cdot P(x_n|Adelie)$$

which is way easier to compute.

Undertsanding how we find that requires some decent knowledge of calculus, so I'm going to try to semplify it as much as possible. If you're really interested in understanding it, you should check other sources.

We are going to focus on only one of the variables, since we can repeat the same process for the others.<br>
We start by removing that "given Adelie" (or the other specie we are considering) by just ... using only the samples from Adelie penguins.

The normal distribution we have assumed the value follows is univocally defined by two values $\overline{x}, \sigma$ which represent the mean and the standard deviation as the function:
$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \ \ e^{- \frac{(x-\overline{x})^2}{2\sigma^2}}$$

notice that those values are ideal, we can't have them. But we can have a good estimation of them through the mean and the standard deviation of the data we have.

Repeating this process we can have a distribution for every element of every pair (specie, variable). We can then calculate the three probalities using:

$$P(x_1,x_2, \dots \ , x_n|Adelie) P(Adelie)$$

and chose the specie with the higher probability.