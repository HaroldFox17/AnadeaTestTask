from simpletransformers.question_answering import QuestionAnsweringModel
# import mlflow
import json
from model_evaluation import get_raw_scores

training_data_path = r"D:\Pet Projects\Anadea\data\training_data_batches.json"
validation_data_path = r"D:\Pet Projects\Anadea\data\validation_data_batches.json"

if __name__ == '__main__':
    # Read the training and validation achieved in the data_preparation.ipynb notebook
    with open(training_data_path, encoding="utf-8") as f:
        training_data = json.load(f)
    with open(validation_data_path, encoding="utf-8") as f:
        validation_data = json.load(f)

    # mlflow.log_params({**train_args, **additional_params})

    model = QuestionAnsweringModel("bert",
                                   r"D:\Pet Projects\Anadea\bert_base_batch_split",
                                   use_cuda=True)

    # model.train_model(training_data[:300], eval_data=validation_data[:100], output_dir='bert_base')
    res = model.predict(validation_data[10:11], n_best_size=5)
    # Calculate and log the loss metric
    em_scores, f1_scores = get_raw_scores(validation_data[:10], res[0])
    # mlflow.log_metric("exact_match_scores", em_scores)
    # mlflow.log_metric("f1_scores", f1_scores)

    # Set a tag that we can use to remind ourselves what this run was for
    # mlflow.set_tag("Training Info", "Basic Bert")
    print(em_scores, f1_scores)
