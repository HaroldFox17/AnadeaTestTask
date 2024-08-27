# SQuAD 2.0 question answering model using finetuning of BERT

This project is dedicated to finetuning a model for SQuAD 2.0 dataset using simpletransformers library. More about SQuAD 2.0 dataset and models used for this problems here:

[https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/). In this project I preprocessed data, trained and evaluated different finetuned models and deployed one of the models using modelbit.

Useful papers about this problem:
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)
- [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/pdf/1906.08237)
- [Ensemble ALBERT on SQuAD 2.0](https://arxiv.org/pdf/2110.09665)


## Project files decription

- __Data_Processing__ notebook that contains formating data to needed format as well as splitting data into train and validation using three different methods
- __model_training__ scipt to train and evaluate model
- __model_evaluation__ script containing functions to evaluate model performance. For this task I used F1 and EM metrics
- __model_prediction__ script to answer questions using local model
- __Model_Deployment__ notebook to deploy models using modelbit

## Interference with the deployed model
You can interact with the deployed model by sending requests to the created endpoint, example using curl given below
```
curl -s -XPOST "https://boglis002.app.modelbit.com/v1/predict_qa_model/latest" -d '{"data": ["What was the cost to society?", "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty prohibiting the hunting of right and gray whales, and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species were relatively rare—and little opposition was raised"]}' | json_pp
```
or using requests library in Python
```
import requests
import json

url = "https://boglis002.app.modelbit.com/v1/predict_qa_model/latest"
headers = {
    'Content-Type': 'application/json'
}
data = {
    "data":  ["What was the cost to society?", "Other legislation followed, including the Migratory Bird Conservation Act of 1929, a 1937 treaty prohibiting the hunting of right and gray whales, and the Bald Eagle Protection Act of 1940. These later laws had a low cost to society—the species were relatively rare—and little opposition was raised"]
}

response = requests.post(url, headers=headers, json=data)
response_json = response.json()

print(json.dumps(response_json, indent=4))
```
## Future improvement steps
- __Proper training__: unfortunately, I couldn't train model on the whole dataset and use more epochs due to hardware limitations, so retraining models properly is needed
- __Different base models__: different base models such as ALBERT, DISTILBERT, XLNET, etc. can be used in the finetuning process to increase the peformance.
- __Model Ensembles__: most of the top competitors use ensembles for their predictions, ALBERT model with ensembles placed 12 places higher than plain single model. I could also use esembles for future improement
