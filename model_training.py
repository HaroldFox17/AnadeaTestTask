from simpletransformers.question_answering import QuestionAnsweringModel
import mlflow
import json
from model_evaluation import get_raw_scores
# import numpy as np

training_data_path = r"D:\Pet Projects\Anadea\data\training_data_within_batches.json"
validation_data_path = r"D:\Pet Projects\Anadea\data\validation_data_within_batches.json"

if __name__ == '__main__':
    mlflow.set_experiment('Bert Finetuning')

    with mlflow.start_run():

        # Read the training and validation achieved in the data_preparation.ipynb notebook
        with open(training_data_path, encoding="utf-8") as f:
            training_data = json.load(f)
        with open(validation_data_path, encoding="utf-8") as f:
            validation_data = json.load(f)

        # Log the hyperparameters
        train_args = {
            'overwrite_output_dir': True,
            "evaluate_during_training": True,
            "max_seq_length": 128,
            "num_train_epochs": 3,  # couldn't make bigger due to hardware limitations
            "evaluate_during_training_steps": 500,
            "save_model_every_epoch": False,
            "save_eval_checkpoints": False,
            "n_best_size": 16,
            "train_batch_size": 8,  # couldn't make batch size bigger due to hardware limitations
            "eval_batch_size": 8
        }

        additional_params = {
            'split_type': 'batch split',
            'model_type': "bert-base-cased"
        }
        mlflow.log_params({**train_args, **additional_params})

        model = QuestionAnsweringModel("bert",
                                       "bert-base-cased",
                                       args=train_args,
                                       use_cuda=True)

        model.train_model(training_data[:300], eval_data=validation_data[:100], output_dir='bert_base_within_batch_split')

        res = model.predict(validation_data[:100], n_best_size=1)[0]

        # Calculate and log the loss metric
        em_scores, f1_scores = get_raw_scores(validation_data[:100], res)
        mlflow.log_metric("exact_match_scores", sum(list(em_scores.values()))/len(list(em_scores.values())))
        mlflow.log_metric("f1_scores", sum(list(f1_scores.values()))/len(list(f1_scores.values())))

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic Bert")
