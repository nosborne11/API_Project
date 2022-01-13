# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model predicts if the expected salary of an individual, given certain details is below or above $50k. Using a logistic decisin tree model 

## Intended Use

Only use for training and development, this is not to be used for real world appilication

## Training Data

The data can be found here: https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data

The evaluation data was composed of a 20% hold out dataset

## Metrics
_Please include the metrics used and your model's performance on those metrics._

On a hold out set the results were as follows:

precision: 0.6183159188690842
recall: 0.6323067253299811
fbeta: 0.6252330640149161


## Ethical Considerations

This was used for training and development, and should not be used in real world situations, the data should not be shared

## Caveats and Recommendations

Expanded to more categories than a binary prediction is recommended
