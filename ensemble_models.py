import os
import argparse
import numpy as np

def load_models_logits(list_of_logits_files, nb_of_models):
    '''
    Save in an array the logits for each logits file given in input.

    '''

    one_model_logits = np.genfromtxt(list_of_logits_files[0][0], delimiter=",")[1:, :]
    print(f"1st model's logits: \n", one_model_logits, "\n")  # to get rid of the header
    nb_of_logits_per_model = len(one_model_logits)
    models_logits_list = [one_model_logits]

    for i in range(nb_of_models):
        if i > 0:
            one_model_logits = np.genfromtxt(list_of_logits_files[i][0], delimiter=",")
            print(f"{i+1}th model's logits: \n", one_model_logits[1:, :], "\n")  # to get rid of the header
            models_logits_list.append(one_model_logits[1:, :])
        
    
    print(f"All the {len(models_logits_list)} models's logits in one array: \n", models_logits_list)
    return models_logits_list, nb_of_logits_per_model


def ensemble_logits(models_logits_list, nb_of_models, nb_of_logits_per_model):
    ensembled_logits = []
    for i in range(nb_of_logits_per_model):
        ensemble_logits_neg = 0
        ensemble_logits_pos = 0
        for model_logits in models_logits_list:
            ensemble_logits_neg += model_logits[i][0]
            ensemble_logits_pos += model_logits[i][1]
        
        ensemble_logits_neg /= nb_of_models
        ensemble_logits_pos /= nb_of_models

        ensembled_logits.append(np.array([ensemble_logits_neg, ensemble_logits_pos]))
    
    ensembled_logits = np.array(ensembled_logits)
    final_predictions = np.argmax(ensembled_logits, axis=1)

    return final_predictions, ensembled_logits

def generate_submission(Y_preds, ensembling_name):
    nb_of_samples=len(Y_preds)
    results = np.zeros((nb_of_samples, 2))

    results[:,0] = np.arange(1, nb_of_samples+1).astype(np.int32)  # save the ids
    results[:,1] = [-1 if elem == 0 else 1 for elem in Y_preds]  # save the test predictions

    ensembling_path = "./" + "Ensembling/"
    os.makedirs(ensembling_path, exist_ok=True)    # create the ensembling folder needed
    final_filename = f"{ensembling_name}-submission.csv"
    np.savetxt(ensembling_path + final_filename, results, fmt="%1d", delimiter=",", header = "Id,Prediction", comments = "")
    
    return final_filename

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-lp", "--logits_path", nargs=1, action='append')
    parser.add_argument('--experiment_name', type=str, default=None, help='Default model name to load')     # "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    config = vars(parser.parse_args())

    if config['experiment_name'] is None:
        config['experiment_name'] = "SomeUnnamedEnsembling"

    
    nb_of_models = len(config['logits_path'])
    print("The logits considered are from ", nb_of_models, " models.\n")
    print("The logits considered are from the following models: \n")
    for i in range(nb_of_models):
        print(config['logits_path'][i][0])

    print("\n")

    models_logits, nb_of_logits_per_model = load_models_logits(config['logits_path'], nb_of_models)
    final_predictions, ensembled_logits = ensemble_logits(models_logits, nb_of_models, nb_of_logits_per_model)

    print("Results of the logits ensembled: \n", ensembled_logits, "\n")
    print("Predictions after ensembling: \n", final_predictions, "\n")

    generate_submission(final_predictions, config['experiment_name'])
