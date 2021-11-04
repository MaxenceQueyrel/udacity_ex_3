# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Maxence Queyrel created this model. It is a logistic regression using the default hyper parameters in scikit-learn 1.0

## Intended Use

The model wants to predict the salary of a population such as there are two classes >50K and <=50k

## Training Data

80% of the dataset (random split), 26048 rows, 108 columns

## Evaluation Data

20% of the dataset (random split), 6513 rows, 108 columns

## Metrics

The metrics used are fbeta, precision and recall

## Ethical Considerations

the data is anonymized

## Caveats and Recommendations

This is an exercise on a benchmark dataset, you can use it to learn basics of data science and MLOps