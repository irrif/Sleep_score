import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from typing import Union

import mlflow
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset

from sklearn.base import BaseEstimator

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split, cross_validate, KFold, StratifiedKFold

from sklearn.metrics import get_scorer_names, mean_squared_error, mean_absolute_error, r2_score

#########################################################################################################################################
######################################################### Statistical functions #########################################################
#########################################################################################################################################

def rmse_score(y_true, y_pred, as_int: bool = True, form: str = ':.4f'):
    """ Compute the RMSE """
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    if as_int:
        return rmse
    return f"{rmse:.4f}"

def mae_score(y_true, y_pred, as_int : bool = True):
    """ Compute the MAE """

    mae = mean_absolute_error(y_true, y_pred)

    if as_int:
        return mae
    return f"{mae:.4f}"

def r2(y_true: np.ndarray = None, y_pred: np.ndarray = None, as_float : bool = True):
    """ Compute the R2 Score """

    score_r2 = r2_score(y_true, y_pred)
    
    if as_float:
        return score_r2
    return f"{score_r2:.4f}"


def adjusted_r2(y_true: np.ndarray = None, X_df: pd.DataFrame = None, r2: float = None, as_float: bool = True):
    """ Compute adjusted R2 Score """

    n = len(y_true)
    p = X_df.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    if as_float:
        return adjusted_r2
    return f"{adjusted_r2:.4f}"


def regression_metrics(y_true, y_pred, X_df):
    """ Return regression metrics RMSE, MAE, R2 and Adjusted_R2 
    
    Parameters :
    ------------
    * y_true : True values
    * y_pred : Predicted values 

    Output : 
    --------
    * tuples : (rmse, mae, r2, adjusted_r2)

    Example :
    ---------
    >>> regression_metrics(y_test, y_pred) -> (0.1, 0.08, 0.85, 0.83)
    """
    rmse = rmse_score(y_true, y_pred)
    mae = mae_score(y_true, y_pred)
    r2_score = r2(y_true, y_pred)
    adjusted_r2_score = adjusted_r2(y_true, X_df, r2_score)
    return rmse, mae, r2_score, adjusted_r2_score


def wilcoxon_mann_whithney(pop, sample):
    """ Compute the U and p_value for Mann Whitney Wilcoxon test """
    U, p = stats.mannwhitneyu(pop, sample)
    return U, p


def assert_p_value(p_value: float = None, thresh: int = 0.05):
    """ Assert if there is a statistical difference based on the p_value and the set threshold """
    print(f"p_value : {p_value:.4f} -> Statistically different") if p_value < thresh else print(f"p_value : {p_value:.4f} -> Not statistically different")


def print_wmh_results(influential_df, whole_df, normal_df, col_name):
    print("High influential VS whole dataset Wilcoxon Mann Whitney test")
    _, mann_p = wilcoxon_mann_whithney(influential_df[col_name], whole_df[col_name])
    print(f"p_value : {mann_p:.4f} -> Statistically different\n") if mann_p < 0.05 else print(f"p_value : {mann_p:.4f} -> Not statistically different\n")

    print('"Normal data" vs whole dataset Wilcoxon Mann Whitney test')
    _, mann_p = wilcoxon_mann_whithney(normal_df[col_name], whole_df[col_name])
    print(f"p_value : {mann_p:.4f} -> Statistically different") if mann_p < 0.05 else print(f"p_value : {mann_p:.4f} -> Not statistically different")


def cramers_corrected_stat(x,y):

    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    result=-1
    if len(x.value_counts())==1 :
        print("First variable is constant")
    elif len(y.value_counts())==1:
        print("Second variable is constant")
    else:   
        conf_matrix=pd.crosstab(x, y)

        if conf_matrix.shape[0]==2:
            correct=False
        else:
            correct=True

        chi2 = stats.chi2_contingency(conf_matrix, correction=correct)[0]

        n = sum(conf_matrix.sum())
        phi2 = chi2/n
        r,k = conf_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        result=np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))
    return round(result,6)


def cramers_matrix(dataf):
    rows= []
    for var1 in dataf:
      col = []
      for var2 in dataf :
        cramers = cramers_corrected_stat(dataf[var1], dataf[var2]) # Cramer's V test
        col.append(round(cramers,2)) # Keeping of the rounded value of the Cramer's V  
      rows.append(col)
    cramers_results = np.array(rows)
    df_resultat = pd.DataFrame(cramers_results, columns = dataf.columns, index =dataf.columns)
    return(df_resultat)

#########################################################################################################################################
########################################################## Utilitary function ###########################################################
#########################################################################################################################################

def transform_cross_val_scores(perf_dict: dict = None, scoring: Union[set, tuple, list] = None) -> dict:
    """
    Transform in the right format scores from cross-validation methods.
    Return the same dictionnary as input with new columns
    
    Parameters :
    ------------
    perf_dict : dict
        Dictionnary containing computed scores

    Returns :
    ---------
    dict
    """

    mapping = {
        'neg_root_mean_squared_error': 'rmse',
        'neg_mean_absolute_error': 'mae',
        'r2': 'r2'
    }

    for score in scoring:
        # Means
        perf_dict[f"train_{mapping[score]}_mean"] = np.abs(perf_dict[f'train_{score}']).mean()
        perf_dict[f"test_{mapping[score]}_mean"] = np.abs(perf_dict[f'test_{score}']).mean()

        # Stddev
        perf_dict[f"train_{mapping[score]}_std"] = np.abs(perf_dict[f'train_{score}']).std()
        perf_dict[f"test_{mapping[score]}_std"] = np.abs(perf_dict[f'test_{score}']).std()

    return perf_dict


def fill_comparison_df(comparison_df: pd.DataFrame = None, metrics: str = None,
                       model_name: str = None, model_metrics: dict = None):
    """
    Fill (inplace) the comparison_df with as key the model_name, and as value a dictionnary that contains
    training and testing means and standard deviation, for the specified metrics.

    Parameters :
    ------------
    metrics : str
        Valid metrics values are :
        * rmse 
        * mae
        * r2

    Returns :
    ---------
    dict

    Examples :
    ----------
    >>> fill_comparison_df(comparison_df, "rmse", "linear_regression", linear_regression_metrics)
    """

    comparison_df[model_name] = {
        f"train_{metrics}_mean": model_metrics[f"train_{metrics}_mean"],
        f"test_{metrics}_mean": model_metrics[f"test_{metrics}_mean"],
        f"train_{metrics}_std": model_metrics[f"train_{metrics}_std"],
        f"test_{metrics}_std": model_metrics[f"test_{metrics}_std"],
    }

#########################################################################################################################################
########################################################## Plotting functions ###########################################################
#########################################################################################################################################
def univariate_continous_EDA(data: pd.DataFrame = None, var: str = None, target: str = 'Score', activity_related: bool = False):
    """ 
    Plots Histogram and Scatterplot versus target variable for continous variables 
    
    Inputs :
    --------
    * data : DataFrame
    * var : Variable to explore
    * target : Target variable of the model
    * activity_related : If the variable is related to activities, remove from the analysis rows where no activities were performed
    """

    if activity_related:
        data = data.loc[data[var] != 0]

    print(data[var].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    bars = sns.histplot(data=data, x=var, ax=axs[0], kde=True)
    sns.scatterplot(data=data, x=var, y=target, ax=axs[1])

    axs[0].set_title(f'{var} repartition')

    axs[1].grid(True, linestyle='--', alpha=0.7)
    axs[1].set_title(f'{var} vs Score')

    for bar in bars.patches:
        height = bar.get_height()
        bars.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'{int(height)}' if height else '',
            ha='center',
            va='bottom'
        )

    plt.tight_layout()
    plt.show()


def continuous_bivariate_scatter(data: pd.DataFrame = None, var_1: str = None, var_2: str = None):
    sns.scatterplot(data=data, x=var_1, y=var_2)

    plt.title(f"{var_1} vs. {var_2}")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_models_metrics(models_metrics: dict = None, models_list: list = None):
    """
    Create a scatter plot comparing train and test RMSE means with color-coding based on model types.

    Inputs :
    --------
    * data : dict
        Dictionary of model metrics.
    * models_list : list
        List of substrings representing model categories (e.g., ["lin_reg", "rf", "xgboost"]).
    """
    # Map model substrings to their full names and colors
    model_mapping = {
        "baseline": ("Linear Regression", "blue"),
        "lin_reg": ("Linear Regression", "blue"),
        "rf": ("Random Forest", "green"),
        "gboost": ("Gradient Boosting", "orange")
    }

    # Assign a color and label to each model based on its name
    def get_model_info(model_name):
        for key in models_list:
            if key in model_name:
                return model_mapping[key]
        return ("Other", "gray")  # Default for unmatched models

    # Extract data and build DataFrame
    models = []
    train_rmse_means = []
    test_rmse_means = []
    categories = []
    colors = []

    for model, metrics in models_metrics.items():
        models.append(model)
        train_rmse_means.append(metrics['train_rmse_mean'])
        test_rmse_means.append(metrics['test_rmse_mean'])
        
        # Get category and color
        category, color = get_model_info(model)
        categories.append(category)
        colors.append(color)

    df = pd.DataFrame({
        'model': models,
        'train_rmse_mean': train_rmse_means,
        'test_rmse_mean': test_rmse_means,
        'category': categories,
        'color': colors
    })

    # Create the scatter plot
    fig = px.scatter(
        df,
        x="test_rmse_mean",
        y="train_rmse_mean",
        text="model",
        color="category",  # Use categories for color
        color_discrete_map={cat: col for cat, col in model_mapping.values()},
        hover_data={
            "model": True,
            "train_rmse_mean": ":.2f",
            "test_rmse_mean": ":.2f"
        },
        title="Train vs Test RMSE Means",
        labels={"test_rmse_mean": "Test RMSE Mean", "train_rmse_mean": "Train RMSE Mean"},
        template="plotly"  # Apply the dark theme
    )

    # Add the diagonal line (y = x)
    x_range = np.linspace(df["test_rmse_mean"].min(), df["test_rmse_mean"].max(), 100)
    fig.add_scatter(
        x=x_range, y=x_range, mode="lines", name="Diagonal (y=x)",
        line=dict(color="black", dash="dash")
    )

    # Update layout for better appearance
    fig.update_traces(marker=dict(size=10), textposition="top center")
    fig.update_layout(
        xaxis=dict(title="Test RMSE Mean"),
        yaxis=dict(title="Train RMSE Mean"),
        showlegend=True,
        legend=dict(title="Model Type")
    )

    fig.show()


def residuals_analysis(df_residuals, x_var) -> None:
    """
    Scatterplot of interest variable vs. Score, with different residuals threshold.
    Histplot of interest variable.

    Inputs :
    --------
    * df_residuals : pd.DataFrame
        DataFrame that contains, the interest variable (x_var), the target, and the residuals.
    * x_var : str
        String that represents the interest variable.

    Returns :
    ---------
    None

    Examples :
    ----------
    >>> residuals_analysis(df_residuals, 'Sleep_duration')
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sns.scatterplot(
        data=df_residuals.loc[abs(df_residuals['train_residuals']) >= 20],
        x=x_var,
        y='Score',
        color='red',
        label='20+ errors',
        ax=axs[0]
    )

    sns.scatterplot(
        data=df_residuals.loc[abs(df_residuals['train_residuals']).between(10, 20, 'left')],
        x=x_var,
        y='Score',
        color='orange',
        label='10 to 20 errors',
        ax=axs[0]
    )

    sns.scatterplot(    
        data=df_residuals.loc[abs(df_residuals['train_residuals']) < 10],
        x=x_var,
        y='Score',
        label='normal errors',
        alpha=0.25,
        ax=axs[0]
    )

    sns.histplot(
        data=df_residuals.loc[abs(df_residuals['train_residuals']) >= 20],
        x=x_var,
        kde=True,
        color='red',
        label='20+ errors',
        ax=axs[1]   
    )

    sns.histplot(
        data=df_residuals.loc[abs(df_residuals['train_residuals']).between(10, 20, 'left')],
        x=x_var,
        kde=True,
        color='orange',
        label='10 to 20 errors',
        ax=axs[1]   
    )

    sns.histplot(
        data=df_residuals.loc[abs(df_residuals['train_residuals']) < 10],
        x=x_var,
        kde=True,
        label='all residuals',
        alpha=0.25,
        ax=axs[1]   
    )

    fig.suptitle(f"Residuals analysis of {x_var}")

    plt.legend()
    plt.show()

#########################################################################################################################################
########################################################### Mlflow functions ############################################################
#########################################################################################################################################

def transform_dataset(whole_df: pd.DataFrame = None, name: str = None, target: str = None):
    """ Transform a Pandas DataFrame to a mlflow compatible DataFrame """

    return mlflow.data.from_pandas(whole_df, name=name, targets=target)


def mlflow_linreg(lr_model: LinearRegression, perf_dict: dict, run_name: str = None,
                  register_dataset: bool = None, dataset = None) -> None:
    """ 
    Save the model and the associated metrics within a mlflow run 
    Can also save the dataset used for training.
        
    Parameters :
    ------------
    * lr_model : RandomForestRegressor
        Linear Regression model
    * perf_dict : dict
        Dictionnary containing metrics
    * run_name : str
        Name of the run
    * register_dataset : bool
        True : register dataset used for training (or whole dataset in case of cross-validation)
    * dataset : PandasDataset
        Pandas DataFrame transformed into a MLflow compatible DataFrame
    """

    if register_dataset and not dataset:
        raise ValueError("The dataset to register is not provided.")

    with mlflow.start_run(run_name=run_name):

        if register_dataset:
            mlflow.log_input(dataset, context='training') 

        mlflow.log_metric("Train_rmse", perf_dict['train_rmse_mean'])
        mlflow.log_metric("Test_rmse", perf_dict['test_rmse_mean'])
        mlflow.log_metric("Train_mae", perf_dict['train_mae_mean'])
        mlflow.log_metric("Test_mae", perf_dict['test_mae_mean'])
        mlflow.log_metric("Train_R2", perf_dict['train_r2_mean'])
        mlflow.log_metric("Test_R2", perf_dict['test_r2_mean'])

        mlflow.sklearn.log_model(lr_model, "Linear_regression")


def mlflow_rforest(rf_model: RandomForestRegressor, perf_dict: dict = None, run_name: str = None,
                   rf_params: dict = None,
                   register_dataset: bool = None, dataset: PandasDataset = None) -> None:
    """ 
    Save the random forest model and the associated parameters and metrics within a mflow run 
    Can also save the dataset used for training.

    Parameters :
    ------------
    * rf_model : RandomForestRegressor
        Random Forest model
    * perf_dict : dict
        Dictionnary containing metrics
    * run_name : str
        Name of the run
    * rf_params : dict
        Dictionnary containing random forest parameters
    * register_dataset : bool
        True : register dataset used for training (or whole dataset in case of cross-validation)
    * dataset : PandasDataset
        Pandas DataFrame transformed into a MLflow compatible DataFrame
    """

    if register_dataset and not dataset:
        raise ValueError("The dataset to register is not provided.")

    with mlflow.start_run(run_name=run_name):
        
        if register_dataset:
            mlflow.log_input(dataset, context='training') 

        # Parameters
        mlflow.log_param("max_depth", rf_params['max_depth'])

        # Metrics
        mlflow.log_metric("Train_rmse", perf_dict['train_rmse_mean'])
        mlflow.log_metric("Test_rmse", perf_dict['test_rmse_mean'])
        mlflow.log_metric("Train_mae", perf_dict['train_mae_mean'])
        mlflow.log_metric("Test_mae", perf_dict['test_mae_mean'])
        mlflow.log_metric("Train_R2", perf_dict['train_r2_mean'])
        mlflow.log_metric("Test_R2", perf_dict['test_r2_mean'])

        mlflow.sklearn.log_model(rf_model, "Random_forest")


def mlflow_gboost(gboost_model: GradientBoostingRegressor, perf_dict: dict = None, run_name: str = None,
                  gboost_params: dict = None,
                  register_dataset: bool = None, dataset = None) -> None:
    
    """ 
    Save the gradient boosting model and the associated parameters and metrics within a mflow run.
    Can also save the dataset used for training.

    Parameters :
    ------------
    * gboost_model : GradientBoostingRegressor
        Gradient Boosting model
    * perf_dict : dict
        Dictionnary containing metrics
    * run_name : str
        Name of the run
    * gboost_params : dict
        Dictionnary containing gradient boosting parameters
    * register_dataset : bool
        True : register dataset used for training (or whole dataset in case of cross-validation)
    * dataset : PandasDataset
        Pandas DataFrame transformed into a MLflow compatible DataFrame
    """

    if not gboost_params or not isinstance(gboost_params, dict):
        raise ValueError("Model parameters are not provided or are not in the correct format (dict).")
    
    if register_dataset and not dataset:
        raise ValueError("The dataset to register is not provided.")

    with mlflow.start_run(run_name=run_name):
        
        if register_dataset:
            mlflow.log_input(dataset, context='training') 

        # Parameters
        mlflow.log_param("learning_rate", gboost_params['learning_rate'])
        mlflow.log_param("n_estimators", gboost_params['n_estimators'])
        mlflow.log_param("max_depth", gboost_params['max_depth'])
        mlflow.log_param("min_samples_leaf", gboost_params['min_samples_leaf'])

        # Metrics
        mlflow.log_metric("Train_rmse", perf_dict['train_rmse_mean'])
        mlflow.log_metric("Test_rmse", perf_dict['test_rmse_mean'])
        mlflow.log_metric("Train_mae", perf_dict['train_mae_mean'])
        mlflow.log_metric("Test_mae", perf_dict['test_mae_mean'])
        mlflow.log_metric("Train_R2", perf_dict['train_r2_mean'])
        mlflow.log_metric("Test_R2", perf_dict['test_r2_mean'])

        mlflow.sklearn.log_model(gboost_model, "Gradient Boosting")


def mlflow_xgboost(xgboost_model: XGBRegressor, perf_dict: dict = None, run_name: str = None,
                   xgboost_params: dict = None,
                   register_dataset: bool = None, dataset = None) -> None:
    """ 
    Save the XGBoost model and the associated parameters and metrics within a mflow run.
    Can also save the dataset used for training.

    Parameters :
    ------------
    * xgboost_model : GradientBoostingRegressor
        XGBoost model
    * perf_dict : dict
        Dictionnary containing metrics
    * run_name : str
        Name of the run
    * xgboost_params : dict
        Dictionnary containing XGBoost parameters
    * register_dataset : bool
        True : register dataset used for training (or whole dataset in case of cross-validation)
    * dataset : PandasDataset
        Pandas DataFrame transformed into a MLflow compatible DataFrame
    """

    # Check that model parameters are provided or in the proper format
    if not xgboost_params or not isinstance(xgboost_params, dict):
        raise ValueError("Model parameters are not provided or are not in the correct format (dict).")
    
    # Check that if register dataset is set to True, a dataset is provided
    if register_dataset and not dataset:
        raise ValueError("The dataset to register is not provided.")

    with mlflow.start_run(run_name=run_name):

        # Register the training dataset
        if register_dataset:
            mlflow.log_input(dataset, context='training') 
        
        # Parameters
        mlflow.log_param("learning_rate", xgboost_params['learning_rate'])
        mlflow.log_param("max_depth", xgboost_params['max_depth'])
        mlflow.log_param("n_estimators", xgboost_params['n_estimators'])

        # Metrics
        mlflow.log_metric("Train_rmse", perf_dict['train_rmse_mean'])
        mlflow.log_metric("Test_rmse", perf_dict['test_rmse_mean'])
        mlflow.log_metric("Train_mae", perf_dict['train_mae_mean'])
        mlflow.log_metric("Test_mae", perf_dict['test_mae_mean'])
        mlflow.log_metric("Train_R2", perf_dict['train_r2_mean'])
        mlflow.log_metric("Test_R2", perf_dict['test_r2_mean'])

        mlflow.set_tag('Scoring method', 'K-Fold')
        
        mlflow.sklearn.log_model(xgboost_model, "XGBoost")


#########################################################################################################################################
####################################################### Model selection functions #######################################################
#########################################################################################################################################

def k_fold_cross_val(X: pd.DataFrame = None, y: pd.Series = None,
                     k_fold: bool = False, stratified: bool = False,
                     n_splits: int = 5, random_state: int = 42,
                     model: BaseEstimator = None, scoring: Union[str, tuple] = None, return_train_score: bool = False) -> dict:
    """
    Returns a dictionnary containing all metrics (scoring) evaluated via cross_validation (default) KFold or StratifiedKFold.

    Parameters :
    ------------
    * X : pd.DataFrame
        Full DataFrame without target variables
    * y : pd.Series
        Target variable Series
    * k_fold : bool
        True : perfroms a K Fold
    * stratified : bool
        True : performs a Stratified K Fold
    * n_splits : int
        Number of folds
    * model : BaseEstimator
        Scikit-learn model to evaluate
    * scoring : str, tuple
        Score to compute
    * return_train_score : bool
        True : return test and training scores
        False : return only test scores
    """

    # For each incorrect score name, return True in the list
    score_check = [score not in get_scorer_names() for score in scoring]

    # Checks for incorrect score names and returns them in ValueError
    if any(score_check):
        raise ValueError(f"{[scoring[idx] for idx, elt in enumerate(score_check) if elt]} are not valid scores.\
                         Run get_scorer_names() to know valid score names.")

    # If no K Fold method
    cv = n_splits

    # Stratified K Fold
    if stratified:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Classic K Fold
    if k_fold:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    return cross_validate(estimator=model, X=X, y=y,
                          cv=cv, scoring=scoring,
                          return_train_score=return_train_score)


def compute_k_fold_cross_val_scores(X: Union[pd.DataFrame, np.ndarray] = None, y: Union[np.ndarray, pd.Series] = None,
                                    model: BaseEstimator = None, random_state: int = 42,
                                    k_fold: bool = False, stratified_k_fold: bool = False,  n_splits: int = 5,
                                    scoring: Union[str, tuple] = None, return_train_score: bool = False,
                                    return_estimator : bool = False) -> dict:
    """
    
    """

    perf_dict = cross_validate(estimator=model, X=X, y=y, 
                               cv=n_splits, 
                               scoring=scoring, return_train_score=return_train_score,
                               return_estimator=return_estimator)

    # K-Fold cross validation
    if k_fold:
        perf_dict = k_fold_cross_val(X=X, y=y, k_fold=True, stratified=False,
                                     n_splits=n_splits, random_state=random_state,
                                     model=model,
                                     scoring=scoring, return_train_score=return_train_score)

    # Stratified K-Fold cross validation
    if stratified_k_fold:
        perf_dict = k_fold_cross_val(X=X, y=y, k_fold=True, stratified=True,
                                     n_splits=n_splits, random_state=random_state,
                                     model=model,
                                     scoring=scoring, return_train_score=return_train_score)
    
    return perf_dict

#########################################################################################################################################
####################################################### Model training functions ########################################################
#########################################################################################################################################

def lin_reg_train_test(X: pd.DataFrame = None, y: pd.Series = None, test_size: int = None,
                       X_train: pd.DataFrame = None, X_test: pd.DataFrame = None, y_train: pd.DataFrame = None, y_test: pd.DataFrame = None, 
                       random_state: int = 42,
                       mlflow_register: bool = False, run_name: str = '',
                       register_dataset: bool = False,  **kwargs) -> tuple[LinearRegression, dict]:
    
    """ Train and return the linear regression + the associated performances.
    The function allows us either to provide the dataset and the target variable, or to provide directly split datasets.
    If mlflow_reg = True, then register the model and the score in mlflow 
    
    Parameters :
    ------------
    * X : pd.DataFrame
        Full DataFrame without target variables
    * y : pd.Series
        Target variable Series
    * X_train, X_test : pd.DataFrame
        train and test dataframes
    * y_train, y_test : np.array, vector, pd.Series
        train and test target values
    * mlflow_register : bool
        True : Register the run with mlflow
    * run_name : str
        run_name of the mlflow run

    Returns :
    --------
    (model, perf_dict)

    Example :
    ---------
    >>> lin_reg_train_test(X_train, X_test, y_train, y_test, mlflow_reg=True, run_name="Base linear regression model")
    >>> lin_reg_train_test(X, y, mlflow_reg=True, run_name="Base linear regression model")
    """

    if X is not None and y is not None:
        if not test_size:
            raise ValueError('When providing a full dataset, a test_size mulst be specified')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    elif any(split is None for split in [X_train, X_test, y_train, y_test]):
        raise ValueError("Either provide full dataset (X and y) or all four split datasets")
    
    if register_dataset:
        full_df = pd.concat([X_train, y_train], axis=1)
        compatible_df = transform_dataset(whole_df=full_df, name='Sleeping score', target='Score')

    perf_dict = {}

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    lr_y_hat_train = lr.predict(X_train)

    perf_dict['Train_rmse'], perf_dict['Train_mae'], perf_dict['Train_R2'], perf_dict['Train_R2_adjusted'] = regression_metrics(y_train, lr_y_hat_train, X_train)

    lr_y_hat_test = lr.predict(X_test)

    perf_dict['Test_rmse'], perf_dict['Test_mae'], perf_dict['Test_R2'], perf_dict['Test_R2_adjusted'] = regression_metrics(y_test, lr_y_hat_test, X_test)

    if mlflow_register:
        if not run_name:
            raise ValueError("mlflow_register is set to True but no run_name has been provided.\
                             \nTo prevent this error input either a run_name or an empty string")
        
        mlflow_linreg(lr_model=lr, perf_dict=perf_dict, run_name=run_name,
                      register_dataset=register_dataset, dataset=compatible_df)

    return lr, perf_dict



def lin_reg_cross_val(X: Union[pd.DataFrame, np.ndarray] = None, y: Union[np.ndarray, pd.Series] = None,
                      random_state: int = 42, k_fold: bool = False, stratified_k_fold: bool = False,
                      n_splits: int = 5, scoring: Union[str, tuple] = None,
                      return_train_score: bool = False, return_estimator: bool = False,
                      mlflow_register: bool = False, register_dataset: bool = False, run_name: str = '', **kwargs) -> dict:
    """
    Train the linear regression and return the associated performances computed via either cross validation, K Fold or Stratified K Fold.

    Parameters :
    ------------
    * X : pd.DataFrame
        - Full DataFrame without target variables
    * y : pd.Series
        - Target variable Series
    * k_fold : bool
        - To perform a K Fold cross validation
    * stratified_k_fold : bool
        - To perform a stratified K Fold cross validation
    * n_splits : int
        - Number of splits for cross validation, or number of folds for K Fold and stratified K Fold
    * scoring : str, tuple
        - metrics to compute  
    * return_train_score : bool
        - True : return test and training scores
        - False : return only test scores
    * return_estimator : bool
        - True : Returns the coefficients of each variable, for each cross-validation
    * mlflow_register : bool
        - True : Register the run with mlflow
    * register_dataset : bool
        - True : Register the training dataset associated to the mlflow run
    * run_name : str
        - run_name of the mlflow run
    """

     # Initialize the linear regression
    linear_regression = LinearRegression()

    perf_dict = compute_k_fold_cross_val_scores(X=X, y=y, model=linear_regression, random_state=random_state, 
                                                k_fold=k_fold, stratified_k_fold=stratified_k_fold,
                                                n_splits=n_splits, scoring=scoring, 
                                                return_train_score=return_train_score, return_estimator=return_estimator)
    
    perf_dict = transform_cross_val_scores(perf_dict=perf_dict, scoring=scoring)
    
    if register_dataset:
        full_df = pd.concat([X, y], axis=1)
        compatible_df = transform_dataset(whole_df=full_df, name='Sleeping score', target='Score')

    # run mlflow 
    if mlflow_register:
        # If no run name provided, raise an Error
        if not run_name:

            raise ValueError("mlflow_register is set to True but no run_name has been provided.\
                             \nTo prevent this error input either a run_name or an empty string")
        
        # Function that executes the mlflow run
        mlflow_linreg(lr_model=linear_regression, perf_dict=perf_dict, run_name=run_name,
                      register_dataset=register_dataset, dataset=compatible_df)
        
    return perf_dict



def random_forest_train_test(X: pd.DataFrame = None, y: pd.Series = None, test_size: int = None,
                             X_train: pd.DataFrame = None, X_test: pd.DataFrame = None, y_train: pd.DataFrame = None, y_test: pd.DataFrame = None,
                             rf_params: dict = None, random_state: int = 42,
                             mlflow_register: bool = False, run_name: str = '', **kwargs) -> tuple[LinearRegression, dict]:
    """ 
    Train and return the random forest + the associated performances.
    The function allows us either to provide the dataset and the target variable, or to provide directly split datasets.
    If mlflow_reg = True then register the model and the score in mlflow 
    
    Parameters :
    ------------
    * X : pd.DataFrame
        Full DataFrame without target variables
    * y : pd.Series
        Target variable Series
    * X_train, X_test : pd.DataFrame
        train and test dataframes
    * y_train, y_test : np.array, vector, pd.Series
        train and test target values
    * rf_params : dict
        Random forest parameters dictionnary with parameter name as key and parameter value as value
    * mlflow_register : bool
        True : Register the run with mlflow
    * run_name : str
        run_name of the mlflow run

    Returns :
    --------
    (model, perf_dict)

    Example :
    ---------
    >>> random_forest_train_test(X_train, X_test, y_train, y_test,
                                 max_depth=3, random_state=42,
                                 mlflow_reg=True, run_name="Base random forest model")
    >>> random_forest_train_test(X, y,
                                 max_depth=3, random_state=42,
                                 mlflow_reg=True, run_name="Base random forest model")
    """

    if X is not None and y is not None:
        if not test_size:
            raise ValueError('When providing a full dataset, a test_size mulst be specified')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    elif any(split is None for split in [X_train, X_test, y_train, y_test]):
        raise ValueError("Either provide full dataset (X and y) or all four split datasets")

    perf_dict = {}

    # Define parameters and train the model
    rf = RandomForestRegressor(max_depth=rf_params['max_depth'], oob_score=True, random_state=random_state)
    rf.fit(X_train, y_train)

    # Compute metrics
    rf_y_hat_train = rf.predict(X_train)

    perf_dict['Train_rmse'], perf_dict['Train_mae'], perf_dict['Train_R2'], perf_dict['rf_adjusted_r2_train'] = regression_metrics(y_train, rf_y_hat_train, X_train)

    rf_y_hat_test = rf.predict(X_test)

    perf_dict['Test_rmse'], perf_dict['Test_mae'], perf_dict['Test_R2'], perf_dict['rf_adjusted_r2_test'] = regression_metrics(y_test, rf_y_hat_test, X_test)

    # run mlflow 
    if mlflow_register:
        if not run_name:
            raise ValueError("mlflow_register is set to True but no run_name has been provided.\
                             \nTo prevent this error input either a run_name or an empty string")
        
        mlflow_rforest(rf_model=rf, perf_dict=perf_dict, run_name=run_name,
                       max_depth=rf_params['max_depth'])

    return rf, perf_dict



def random_forest_cross_val(X: Union[pd.DataFrame, np.ndarray] = None, y: Union[np.ndarray, pd.Series] = None,
                            rf_params: dict = None, random_state: int = 42,
                            k_fold: bool = False, stratified_k_fold: bool = False,
                            n_splits: int = 5, scoring: Union[str, tuple] = None, return_train_score: bool = False,
                            mlflow_register: bool = False, register_dataset: bool = False, run_name: str = '', **kwargs) -> dict:
    """
    Train the random forest and return the associated performances computed via either cross validation, K Fold or Stratified K Fold.

    Parameters :
    ------------
    * X : pd.DataFrame
        Full DataFrame without target variables
    * y : pd.Series
        Target variable Series
    * rf_params : dict
        Random forest parameters dictionnary with parameter name as key and parameter value as value
    * k_fold : bool
        To perform a K Fold cross validation
    * stratified_k_fold : bool
        To perform a stratified K Fold cross validation
    * n_splits : int
        Number of splits for cross validation, or number of folds for K Fold and stratified K Fold
    * scoring : str, tuple
        metrics to compute  
    * return_train_score : bool
        True : return test and training scores
        False : return only test scores
    """

    # Initialize the random forest with specified parameters if they exists, otherwise default parameters
    if rf_params:
        random_forest = RandomForestRegressor(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_leaf=rf_params['min_samples_leaf'],
            oob_score=True,
            random_state=random_state)
    else:
        random_forest = RandomForestRegressor(
            oob_score=True,
            random_state=random_state
        )

    perf_dict = compute_k_fold_cross_val_scores(X=X, y=y, model=random_forest,
                                                random_state=random_state, k_fold=k_fold, stratified_k_fold=stratified_k_fold,
                                                n_splits=n_splits, scoring=scoring, return_train_score=return_train_score)
    
    perf_dict = transform_cross_val_scores(perf_dict=perf_dict, scoring=scoring)
    
    if register_dataset:
        full_df = pd.concat([X, y], axis=1)
        compatible_df = transform_dataset(whole_df=full_df, name='Sleeping score', target='Score')

    # run mlflow 
    if mlflow_register:
        # If no run name provided, raise an Error
        if not run_name:

            raise ValueError("mlflow_register is set to True but no run_name has been provided.\
                             \nTo prevent this error input either a run_name or an empty string")
        
        # Function that executes the mlflow run
        mlflow_rforest(rf_model=random_forest, perf_dict=perf_dict, run_name=run_name,
                       rf_params=rf_params,
                       register_dataset=register_dataset, dataset=compatible_df)
        
    return perf_dict



def gradient_boosting_train_test(X: pd.DataFrame = None, y: pd.Series = None, test_size: int = None,
                                 X_train: pd.DataFrame = None, X_test: pd.DataFrame = None, y_train: pd.DataFrame = None, y_test: pd.DataFrame = None,
                                 gboost_params: dict = None, random_state: int = 42,
                                 mlflow_register: bool = False, run_name: str = '', register_dataset: bool = False,
                                 return_train_test: bool = False, **kwargs) -> tuple[np.ndarray, np.ndarray, LinearRegression, dict]:
    
    """ 
    Train and return the gradient boosting model + the associated performances.
    The function allows us either to provide the dataset and the target variable, or to provide directly split datasets.
    If mlflow_reg = True then register the model and the score in mlflow.
    If register_dataset = True then register the training dataset in the mlflow run.
    
    Parameters :
    ------------
    * X : pd.DataFrame
        Full DataFrame without target variables
    * y : pd.Series
        Target variable Series
    * test_size : int
        Test size for the train_test_split function
    * X_train, X_test : pd.DataFrame
        train and test dataframes
    * y_train, y_test : np.array, vector, pd.Series
        train and test target values
    * gboost_params : dict
        Gradient boosting parameters dictionnary with parameter name as key and parameter value as value
    * mlflow_register : bool
        True : Register the run with mlflow
    * run_name : str
        run_name of the mlflow run
    * register_dataset : bool
        True : Register the training dataset associated to the mlflow run
    * return_train_test : bool
        True : Return training and testing predictions

    Returns :
    --------
    (model, perf_dict)

    Example :
    ---------
    >>> gradient_boosting_train_test(X_train, X_test, y_train, y_test,
                                     gboost_params=gboost_params_dict, random_state=42,
                                     mlflow_reg=True, run_name="Base gradient boosting model")
    >>> gradient_boosting_train_test(X, y,
                                     gboost_params=gboost_params_dict, random_state=42,
                                     mlflow_reg=True, run_name="Base gradient boosting model")
    """
    
    # Check that provided data is usable.
    if X is not None and y is not None:

        if not test_size:

            raise ValueError('When providing a full dataset, a test_size mulst be specified')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    # Check if at least one part of splitted dataset is Null
    elif any(split is None for split in [X_train, X_test, y_train, y_test]):
        raise ValueError("Either provide full dataset (X and y) or all four split datasets")
    
    # Create the dataset to register with both features and target.
    if register_dataset:
        full_df = pd.concat([X_train, y_train], axis=1)
        compatible_df = transform_dataset(whole_df=full_df, name='Sleeping score', target='Score')

    perf_dict = {}

    # Check that model parameters are present
    if not gboost_params or not isinstance(gboost_params, dict):
        raise ValueError("Model parameters are not provided or are not in the correct format (dict)")

    # Define parameters and train the model
    gboost = GradientBoostingRegressor(
        learning_rate = gboost_params['learning_rate'],
        n_estimators = gboost_params['n_estimators'],
        max_depth = gboost_params['max_depth'],
        # min_samples_leaf = gboost_params['min_samples_leaf'],
        verbose = gboost_params['verbose'],
        random_state = random_state
)
    gboost.fit(X_train, y_train)
    
    # Compute metrics
    gboost_y_hat_train = gboost.predict(X_train)

    perf_dict['gboost_rmse_train'], perf_dict['gboost_mae_train'], perf_dict['gboost_r2_train'], perf_dict['gboost_adjusted_r2_train'] = regression_metrics(y_train, gboost_y_hat_train, X_train)

    gboost_y_hat_test = gboost.predict(X_test)

    perf_dict['gboost_rmse_test'], perf_dict['gboost_mae_test'], perf_dict['gboost_r2_test'], perf_dict['gboost_adjusted_r2_test'] = regression_metrics(y_test, gboost_y_hat_test, X_test)

    # run mlflow 
    if mlflow_register:
        # If no run name provided, raise an Error
        if not run_name:

            raise ValueError("mlflow_register is set to True but no run_name has been provided.\
                             \nTo prevent this error input either a run_name or an empty string")
        
        # Function that executes the mlflow run
        mlflow_gboost(gboost_model=gboost, perf_dict=perf_dict, run_name=run_name,
                      gboost_params=gboost_params,
                      register_dataset=register_dataset, dataset=compatible_df)

    # return train prediction, test prediction, model, scores
    if return_train_test:
        return gboost_y_hat_train, gboost_y_hat_test, gboost, perf_dict
    
    # Only return model and scores
    return gboost, perf_dict



def gradient_boosting_cross_val(X: Union[pd.DataFrame, np.ndarray] = None, y: Union[np.ndarray, pd.Series] = None,
                                gboost_params: dict = None, random_state: int = 42,
                                k_fold: bool = False, stratified_k_fold: bool = False,
                                n_splits: int = 5, scoring: Union[str, tuple] = None, return_train_score: bool = False,
                                mlflow_register: bool = False, register_dataset: bool = False, run_name: str = '', **kwargs) -> dict:
    """
    Train the gradient boosting and return the associated performances computed via either cross validation, K Fold or Stratified K Fold.

    Parameters :
    ------------
    * X : pd.DataFrame
        Full DataFrame without target variables
    * y : pd.Series
        Target variable Series
    * gboost_params : dict
        Gradient Boosting parameters dictionnary with parameter name as key and parameter value as value
    * k_fold : bool
        To perform a K Fold cross validation
    * stratified_k_fold : bool
        To perform a stratified K Fold cross validation
    * n_splits : int
        Number of splits for cross validation, or number of folds for K Fold and stratified K Fold
    * scoring : str, tuple
        metrics to compute  
    * return_train_score : bool
        True : return test and training scores
        False : return only test scores
    """

    # Initialize the gradient boosting with specified parameters if they exists, otherwise default parameters
    if gboost_params:
        gboost = GradientBoostingRegressor(
            learning_rate = gboost_params['learning_rate'],
            n_estimators = gboost_params['n_estimators'],
            max_depth = gboost_params['max_depth'],
            min_samples_leaf = gboost_params['min_samples_leaf'],
            verbose = gboost_params['verbose'],
            random_state = random_state
        )
    else:
        gboost = GradientBoostingRegressor(
            random_state=random_state
        )
    
    perf_dict = compute_k_fold_cross_val_scores(X=X, y=y, model=gboost,
                                                random_state=random_state, k_fold=k_fold, stratified_k_fold=stratified_k_fold,
                                                n_splits=n_splits, scoring=scoring, return_train_score=return_train_score)
    
    perf_dict = transform_cross_val_scores(perf_dict=perf_dict, scoring=scoring)
    
    if register_dataset:
        full_df = pd.concat([X, y], axis=1)
        compatible_df = transform_dataset(whole_df=full_df, name='Sleeping score', target='Score')
    else:
        compatible_df = None

    # run mlflow 
    if mlflow_register:
        # If no run name provided, raise an Error
        if not run_name:

            raise ValueError("mlflow_register is set to True but no run_name has been provided.\
                             \nTo prevent this error input either a run_name or an empty string")
        
        # Function that executes the mlflow run
        mlflow_gboost(gboost_model=gboost, perf_dict=perf_dict, run_name=run_name,
                      gboost_params=gboost_params,
                      register_dataset=register_dataset, dataset=compatible_df)

    return perf_dict


def xgboost_train_test(X: pd.DataFrame = None, y: pd.Series = None, test_size: int = None,
                       X_train: pd.DataFrame = None, X_test: pd.DataFrame = None, y_train: pd.DataFrame = None, y_test: pd.DataFrame = None,
                       xgboost_params: dict = None, random_state: int = 42,
                       mlflow_register: bool = False, run_name: str = '', register_dataset: bool = False,
                       return_train_test: bool = False, **kwargs) -> tuple[np.ndarray, np.ndarray, LinearRegression, dict]:
    
    """ 
    Train and return the XGBoost model + the associated performance.
    The function allows us either to provide the dataset and the target variable, or to provide directly split datasets.
    If mlflow_reg = True then register the model and the score in mlflow.
    If register_dataset = True then register the training dataset in the mlflow run.
    
    Parameters :
    ------------
    * X : pd.DataFrame
        Full DataFrame without target variables
    * y : pd.Series
        Target variable Series
    * test_size : int
        Test size for the train_test_split function
    * X_train, X_test : pd.DataFrame
        train and test dataframes
    * y_train, y_test : np.array, vector, pd.Series
        train and test target values
    * xgboost_params : dict
        XGBoost parameters dictionnary with parameter name as key and parameter value as value
    * mlflow_register : bool
        True : Register the run with mlflow
    * run_name : str
        run_name of the mlflow run
    * register_dataset : bool
        True : Register the training dataset associated to the mlflow run
    * return_train_test : bool
        True : Return training and testing predictions

    Returns :
    --------
    (model, perf_dict)
    (train_predictions, test_predictions, model, perf_dict)

    Example :
    ---------
    >>> xgboost_train_test(X_train, X_test, y_train, y_test,
                           xgboost_params=xgboost_params_dict, random_state=42,
                           mlflow_reg=True, run_name="Base XGBoost model")
    >>> xgboost_train_test(X, y,
                           xgboost_params=xgboost_params_dict, random_state=42,
                           mlflow_reg=True, run_name="Base XGBoost model, register_dataset=True")
    """
    
    # Check that provided data is usable.
    if X is not None and y is not None:

        if not test_size:

            raise ValueError('When providing a full dataset, a test_size mulst be specified')
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
    # Check if at least one part of splitted dataset is Null
    elif any(split is None for split in [X_train, X_test, y_train, y_test]):
        raise ValueError("Either provide full dataset (X and y) or all four split datasets")
    
    # Create the dataset to register with both features and target.
    if register_dataset:
        full_df = pd.concat([X_train, y_train], axis=1)
        compatible_df = transform_dataset(whole_df=full_df, name='Sleeping score', target='Score')
    else:
        compatible_df = None

    perf_dict = {}

    # Check that model parameters are present
    if not xgboost_params or not isinstance(xgboost_params, dict):
        raise ValueError("Model parameters are not provided or are not in the correct format (dict)")

    # Define parameters and train the model
    xgboost = XGBRegressor(
        learning_rate = xgboost_params['learning_rate'],
        n_estimators = xgboost_params['n_estimators'],
        max_depth = xgboost_params['max_depth'],
        verbosity = xgboost_params['verbosity'],
        n_jobs = xgboost_params['n_jobs'],
        random_state = random_state
    )

    xgboost.fit(X_train, y_train)
    
    # Compute metrics
    xgboost_y_hat_train = xgboost.predict(X_train)

    perf_dict['xgboost_rmse_train'], perf_dict['xgboost_mae_train'], perf_dict['xgboost_r2_train'], perf_dict['xgboost_adjusted_r2_train'] = regression_metrics(y_train, xgboost_y_hat_train, X_train)

    xgboost_y_hat_test = xgboost.predict(X_test)

    perf_dict['xgboost_rmse_test'], perf_dict['xgboost_mae_test'], perf_dict['xgboost_r2_test'], perf_dict['xgboost_adjusted_r2_test'] = regression_metrics(y_test, xgboost_y_hat_test, X_test)

    # Mlflow part
    if mlflow_register:
        # If no run name provided, raise an Error
        if not run_name:
            raise ValueError("mlflow_register is set to True but no run_name has been provided.\
                             \nTo prevent this error input either a run_name or an empty string")
        
        # Function that executes the mlflow run
        mlflow_xgboost(xgboost_model=xgboost, perf_dict=perf_dict, run_name=run_name,
                       xgboost_params=xgboost_params,
                       register_dataset=register_dataset, dataset=compatible_df)
        
    # return train prediction, test prediction, model, scores
    if return_train_test:
        return xgboost_y_hat_train, xgboost_y_hat_test, xgboost, perf_dict
    
    # Only return model and scores
    return xgboost, perf_dict



def xgboost_cross_val(X: Union[pd.DataFrame, np.ndarray] = None, y: Union[np.ndarray, pd.Series] = None,
                      xgboost_params: dict = None, random_state: int = 42,
                      k_fold: bool = False, stratified_k_fold: bool = False,
                      n_splits: int = 5, scoring: Union[str, tuple] = None, return_train_score: bool = False,
                      mlflow_register: bool = False, register_dataset: bool = False, run_name: str = '', **kwargs) -> dict:
    """
    Train the XGBoost model and return the associated performances computed via either cross validation, K Fold or Stratified K Fold.
    
    Parameters :
    ------------
    * X : pd.DataFrame
        Full DataFrame without target variables
    * y : pd.Series
        Target variable Series
    * xgboost_params : dict
        XGBoost parameters dictionnary with parameter name as key and parameter value as value
    * k_fold : bool
        To perform a K Fold cross validation
    * stratified_k_fold : bool
        To perform a stratified K Fold cross validation
    * n_splits : int
        Number of splits for cross validation, or number of folds for K Fold and stratified K Fold
    * scoring : str, tuple
        metrics to compute  
    * return_train_score : bool
        True : return test and training scores
        False : return only test scores
    """

    # Initialize the xgboost with specified parameters if they exists, otherwise default parameters
    if xgboost_params:
        xgboost = XGBRegressor(
            learning_rate = xgboost_params['learning_rate'],
            n_estimators = xgboost_params['n_estimators'],
            max_depth = xgboost_params['max_depth'],
            verbosity = xgboost_params['verbosity'],
            n_jobs = xgboost_params['n_jobs'],
            random_state = random_state
        )
    else:
        xgboost = XGBRegressor(
            random_state=random_state
        )

    perf_dict = compute_k_fold_cross_val_scores(X=X, y=y, model=xgboost,
                                                random_state=random_state, k_fold=k_fold, stratified_k_fold=stratified_k_fold,
                                                n_splits=n_splits, scoring=scoring, return_train_score=return_train_score)
    
    perf_dict = transform_cross_val_scores(perf_dict=perf_dict, scoring=scoring)
    
    if register_dataset:
        full_df = pd.concat([X, y], axis=1)
        compatible_df = transform_dataset(whole_df=full_df, name='Sleeping score', target='Score')
    else:
        compatible_df = None

    # run mlflow 
    if mlflow_register:
        # If no run name provided, raise an Error
        if not run_name:

            raise ValueError("mlflow_register is set to True but no run_name has been provided.\
                             \nTo prevent this error input either a run_name or an empty string")
        
        # Function that executes the mlflow run
        mlflow_xgboost(xgboost_model=xgboost, perf_dict=perf_dict, run_name=run_name,
                       xgboost_params=xgboost_params,
                       register_dataset=register_dataset, dataset=compatible_df)

    return perf_dict