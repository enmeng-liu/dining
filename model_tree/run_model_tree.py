"""

 model_tree.py  (author: Anson Wong / git: ankonzoid)

 Given a classification/regression model, this code builds its model tree.

"""
import os, pickle, csv
from src.ModelTree import ModelTree
from src.utils import load_csv_data, cross_validate

import warnings
warnings.filterwarnings("ignore")

def main():
    # ====================
    # Settings
    # ====================
    mode = "regr"  # "clf" / "regr"
    save_model_tree = True  # save model tree?
    save_model_tree_predictions = False  # save model tree predictions/explanations?
    cross_validation = False  # cross-validate model tree?

    # ====================
    # Load data
    # ====================
    # data_csv_data_filename = os.path.join("data", "/home/congee/Documents/Project_preparation/train.csv")
    data_csv_data_filename = "/home/congee/Documents/Project_preparation/train.csv"
    X, y, header = load_csv_data(data_csv_data_filename, mode=mode, verbose=True)

    # *********************************************
    #
    # Insert your models here!
    #
    # All models must have the following class instantiations:
    #
    #   fit(X, y)
    #   predict(X)
    #   loss(X, y, y_pred)
    #
    # Below are some ready-for-use regression models:
    #
    #   mean regressor  (models/mean_regr.py)
    #   linear regressor  (models/linear_regr.py)
    #   logistic regressor  (lmodels/ogistic_regr.py)
    #   support vector machine regressor  (models/svm_regr.py)
    #   decision tree regressor (models/DT_sklearn_regr.py)
    #   neural network regressor (models/DT_sklearn_regr.py)
    #
    # as well as some classification models:
    #
    #   modal classifier (models/modal_clf.py)
    #   decision tree classifier (models/DT_sklearn_clf.py)
    #
    # *********************************************
    from models.mean_regr import mean_regr
    from models.linear_regr import linear_regr
    from models.logistic_regr import logistic_regr
    from models.svm_regr import svm_regr
    from models.DT_sklearn_regr import DT_sklearn_regr

    from models.modal_clf import modal_clf
    from models.DT_sklearn_clf import DT_sklearn_clf

    # Choose model
    model = linear_regr()

    # Build model tree
    model_tree = ModelTree(model, max_depth=5, min_samples_leaf=3,
                           search_type="greedy", n_search_grid=100)

    # ====================
    # Train model tree
    # ====================
    print("Training model tree with '{}'...".format(model.__class__.__name__))
    model_tree.fit(X, y, verbose=True)
    y_pred = model_tree.predict(X)
    explanations = model_tree.explain(X, header)
    loss = model_tree.loss(X, y, y_pred)
    print(" -> loss_train={:.6f}\n".format(loss))
    model_tree.export_graphviz(os.path.join("output", "model_tree"), header,
                               export_png=True, export_pdf=False)

    # ====================
    # Save model tree results
    # ====================
    if save_model_tree:
        model_tree_filename = os.path.join("output", "model_tree.p")
        print("Saving model tree to '{}'...".format(model_tree_filename))
        pickle.dump(model, open(model_tree_filename, 'wb'))

    if save_model_tree_predictions:
        predictions_csv_filename = os.path.join("output", "model_tree_pred.csv")
        print("Saving mode tree predictions to '{}'".format(predictions_csv_filename))
        with open(predictions_csv_filename, "w") as f:
            writer = csv.writer(f)
            field_names = ["x", "y", "y_pred", "explanation"]
            writer.writerow(field_names)
            for (x_i, y_i, y_pred_i, exp_i) in zip(X, y, y_pred, explanations):
                field_values = [x_i, y_i, y_pred_i, exp_i]
                writer.writerow(field_values)

    # ====================
    # Cross-validate model tree
    # ====================
    if cross_validation:
        cross_validate(model_tree, X, y, kfold=5, seed=1)

    import pandas as pd
    import numpy as np
    file_name = "/home/congee/Documents/Project_preparation/test.csv"
    data = pd.read_csv(file_name, header=0, sep=',')
    del data['id']
    del data['scale']
    del data['random_state']
    data['l2_ratio'] = 0
    data['l1_ratio'][data['penalty'] == 'none'] = 0
    data['l2_ratio'][data['penalty'] == 'none'] = 0
    data['l1_ratio'][data['penalty'] == 'l1'] = 1
    data['l2_ratio'][data['penalty'] == 'l1'] = 0
    data['l1_ratio'][data['penalty'] == 'l2'] = 0
    data['l2_ratio'][data['penalty'] == 'l2'] = 1
    data['l2_ratio'][data['penalty'] == 'elasticnet'] = 1 - data['l1_ratio'][data['penalty'] == 'elasticnet']
    data['n_jobs'][data['n_jobs'] == -1] = 16
    del data['penalty']
    X_test = np.array(data)
    Y_test = model_tree.predict(X_test)

    ans = pd.DataFrame(Y_test)
    ans.to_csv('/home/congee/Documents/Project_preparation/res_5.csv', sep=',')

# Driver
if __name__ == "__main__":
    main()