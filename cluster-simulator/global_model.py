# %%
from utils import load_workload, load_config, get_num_task_per_node, random_forest, logistic_regression
import matplotlib.pyplot as plt
import pandas as pd
from simulate import scale_workload
import logging
import joblib
from sklearn import tree
import matplotlib.pyplot as plt
import os.path

data = load_workload('workload')
config = load_config()

ITERATION = 16
WORKLOAD = 8


def iterative_data_generation(iteration_count, workload_count):
    """
    A function to generate data iteratively and store it in a DataFrame.

    Args:
    iteration_count (int): The number of iterations for each workload.
    workload_count (int): The number of workloads.

    Returns:
    None. Writes the data to a CSV file.
    """

    result_dataframe = pd.DataFrame()
    configuration = load_config()

    for workload in range(1, workload_count):
        workload_data = load_workload(f'workload{workload}')
        for iteration in range(1, iteration_count):
            configuration['Initial_Node_Count'] = iteration
            scaled_data = scale_workload(workload_data, configuration)
            result_dataframe = pd.concat([result_dataframe, pd.DataFrame([scaled_data])], ignore_index=True)
    result_dataframe.to_csv('ml_model/model_output.csv', index=False)


def train_model(model_selection):
    """
    A function to train a machine learning model based on a specified model selection.

    Args:
    model_selection (str): The type of model to train. Options: 'lr' for logistic regression and 'rf' for random forest.

    Returns:
    None. Saves the trained model to a .joblib file.
    """

    data_frame = pd.read_csv('ml_model/model_output.csv')

    target = data_frame['Node_Count']
    predictors = data_frame.drop(['Node_Count', 'No_of_Completed_Tasks', 'No_of_Rejected_Tasks'], axis=1)
    predictors_list = predictors.values.tolist()
    target_list = target.values.tolist()

    logging.info(f"Length of predictors: {len(predictors_list)}")
    logging.info(f"Length of target: {len(target_list)}")

    if model_selection == 'lr':
        logistic_regression_model = logistic_regression(predictors_list, target_list)
        joblib.dump(logistic_regression_model, 'ml_model/logistic_regression_model.joblib')
    elif model_selection == 'rf':
        random_forest_model = random_forest(predictors_list, target_list)
        joblib.dump(random_forest_model, 'ml_model/random_forest_model.joblib')



def calc_completion_time(task):
    """
    A function to calculate the completion time of a task.

    Args:
    task (Task): The task to calculate the completion time for.

    Returns:
    float. The completion time of the task.
    """
    return task.num_tasks * config['Time_To_Complete_Task'] / get_num_task_per_node(task, config)


def calc_overall_utilization(task):
    """
    A function to calculate the overall utilization of a task.

    Args:
    task (Task): The task to calculate the overall utilization for.

    Returns:
    float. The overall utilization of the task.
    """

    return max(task.cpu, task.memory, task.network) * get_num_task_per_node(task, config)


def visulize_tree():
    """
    A function to visualize a single tree from the random forest model.

    Args:
    None.

    Returns:
    None. Saves the visualization to a .png file.
    """

    # check if the model is saved
    if not os.path.isfile('ml_model/random_forest_model.joblib'):
        print("Model not found")
        return
    
    # Load the model
    loaded_model = joblib.load('ml_model/random_forest_model.joblib')

    # Extract single tree
    estimator = loaded_model.estimators_[50]

    # Visualize the tree
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=800)
    tree.plot_tree(estimator,
                filled = True)
    fig.savefig('rf_individualtree.png')


if __name__ == '__main__':
    X_data = load_workload('workload7')

    input_lst = [[80, 80, calc_completion_time(data), data.num_tasks] for data in X_data]

    loaded_model = joblib.load('ml_model/random_forest_model.joblib')
    prd_node_lst = loaded_model.predict(input_lst)
    prd_node_lst = [int(i) for i in prd_node_lst]

    print(prd_node_lst)