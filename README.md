# LogisticRegression Experiment

While working through Chapter 3 Feature Selection II - Selecting for Model Accuracy of the [Dimensionality Reduction in Python](https://app.datacamp.com/learn/courses/dimensionality-reduction-in-python) course on [datacamp.com](https://datacamp.com) I was unable to fully replicate the site's results on my local machine and wanted to understand why.

The site and I were getting different results. The site would consistently get:

```sh
{'pregnant': 5, 'glucose': 1, 'diastolic': 6, 'triceps': 3, 'insulin': 4, 'bmi': 1, 'family': 2, 'age': 1}
Index(['glucose', 'bmi', 'age'], dtype='object')
80.6% accuracy on test set.
```

While my results vary but almost always include the `'pregnant'` feature unless I drop it from the dataset.

According to my experiments below, the site and I were producing identical correlation matrices and our heatmaps were, not surprisingly, the same as well.

Interestingly, if I don't increase the `max_iter` parameter I would get the following *after* my results:

```sh
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
```

The value I needed to set for `max_iter` was not constant but I never saw the error with a value >= 200.

My first thought was that perhaps the default solver has changed was different.

On the site:

```sh
In [16]: print(LogisticRegression().solver)
lbfgs
```

and on my machine:

```sh
>>> print(LogisticRegression().solver)
lbfgs
```

I then checked the version of scikit-learn.

The site's version:

```sh
In [17]: import sklearn
In [18]: print('sklearn: {}'. format(sklearn. __version__))
sklearn: 1.0
```

and on my machine:

```sh
>>> import sklearn
>>> print('sklearn: {}'. format(sklearn. __version__))
sklearn: 1.3.2
```

My next thought was to try installing scikit-learn v1.0 on my machine to see if I can reproduce the site's results. This, however, turned out to be more involved than I expected due to dependency issues. Instead, I built a separate env with numpy v1.19.5, pandas v1.3.4, scikit-learn v1.0, and Python v3.9.7 to mirror the site's environment. The result is this repo.
