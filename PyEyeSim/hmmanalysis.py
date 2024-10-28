"""
This module provides functions for simulating and evaluating Gaussian Hidden Markov Models (GHMMs) used to analyze eye fixation data.

### Data Acquisition and Preprocessing

* **get_data(self, stim, tolerance=20)**
    - Retrieves fixation data for a specific stimulus (`stim`) within a tolerance level (`tolerance`).
    - Returns the X-axis (`XX`), Y-axis (`YY`), and fixation length data (`list_lengths`).

* **fixation_sequence(lengths, method='random')**
    - Generates a sequence of fixation lengths based on a provided list (`lengths`) and chosen method (`method`).
    - Supported methods: 'random', 'average', 'max' or specific integer values.

### Model Fitting and Simulation

* **models_fit(X, Y, models, lengths)**
    - Fits the provided GaussianHMM models (`models`) to the eye fixation data (`X`, `Y`) considering the fixation sequence lengths (`lengths`).

* **model_def(X, Y, list_lengths, n_components_list, covariance_type='full')**
    - Defines GaussianHMM models with different numbers of components (`n_components_list`) based on the input data and list of fixation lengths.
    - Utilizes the `hmmlearn.hmm.GaussianHMM` class for model creation.

* **data_simulation(models, lengths)**
    - Simulates eye fixation data based on the fitted models (`models`) and provided fixation sequence lengths (`lengths`).

### Model Evaluation

* **score_calculation(method, model, data, lengths, lda_model=None)**
    - Calculates different evaluation scores for a model based on the chosen method (`method`).
    - Supported methods: 'score' (log-likelihood), 'bic', 'aic', 'lda'.

* **calculate_starting_likelihood(X, Y, list_lengths, n_components_list, covariance_type, starting_tests=10, only_bic=False)**
    - Performs initial evaluation by fitting models with different numbers of components and calculates various scores (log-likelihood, BIC, AIC) for each.
    - Allows specifying the number of starting tests (`starting_tests`) and whether to calculate only BIC (`only_bic`).

* **likelihood_matrix(models, simulated_data, lengths, n_components_list, evaluation='score', lda_models=None)**
    - Creates a likelihood matrix based on the evaluation criteria (`evaluation`).
    - It can evaluate models using score, BIC, AIC, LDA, or all of them.

### Visualization

* **plot_functions** (functions not shown here, but named descriptively, e.g., `plot_likelihood_matrix`, `plot_simulated_data`)
    - Generate various plots to visualize the results (likelihood matrices, simulated data, model summaries).

* **plot_pipeline(self, models, stimuli, iteration, simulated_X, simulated_Y, n_components_list, new_list_len, likelihood_mat, evaluation, Summary=True)**
    - Combines functionalities for plotting the entire pipeline's results.
    - It can display simulated data, likelihood matrices (depending on evaluation criteria), and model summaries.

### Main Pipeline Function

* **models_pipeline(self, stim, n_components_list, iteration=1, tollerance=20, simulation_type='random', evaluation='all', covariance_type='full', starting_tests=10, only_starting=False, only_bic=False, only_best=False, threshold=0.5)**
    - The main entry point for the pipeline.
    - Performs the following steps:
        1. Retrieves fixation data for the stimulus (`stim`).
        2. Defines GaussianHMM models with different numbers of components (`n_components_list`).
        3. Optionally performs initial evaluation with multiple starting tests (`starting_tests`).
        4. Simulates eye fixation data based on the models and chosen simulation type (`simulation_type`).
        5. Evaluates the models using the specified criteria (`evaluation`).
        6. Optionally selects the best models based on concordance (`only_best`) using a threshold (`threshold`).
        7. Returns the fitted models (`models`), simulated data (`simulated_data`), and potentially the likelihood matrix (`likelihood_mat`).

* **GaussianHMMPipeline(self, stim, n_components_list, iteration=1, tollerance=20, simulation_type='random', evaluation='all', covariance_type='full', Summary=True, starting_tests=10, only_starting=False, only_bic=False, only_best=False, threshold=0.5)**
    - A more user-friendly wrapper function for the pipeline.
    - Calls `models_pipeline` and optionally plots the results based on the `Summary` parameter.
   
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import hmmlearn.hmm  as hmm
from IPython.utils import io
import copy

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_data(self, stim, group=-1, tolerance=20, subject=-1, remove_subj=False):
    """
    Get the fixation data for a specific stimulus.

    Parameters
    ----------
    stim : int
        The stimulus number.

    tolerance : int, default=20
    
    Returns
    -------
    np.array
        The X-axis fixation data.

    np.array
        The Y-axis fixation data.

    list
        The list of fixation lengths for each subject.
    """
    
    XX, YY, list_lengths = self.DataArrayHmm(stim, group, tolerance=tolerance, verb=False)

    if subject != -1:
        if remove_subj:
            try:
                lengths_sum = np.cumsum(list_lengths)
                if subject == 0:
                    XX = XX[list_lengths[0]:]
                    YY = YY[list_lengths[0]:]
                else:
                    XX = np.concatenate([XX[:lengths_sum[subject-1]], XX[lengths_sum[subject]:]])
                    YY = np.concatenate([YY[:lengths_sum[subject-1]], YY[lengths_sum[subject]:]])
                list_lengths = np.concatenate([list_lengths[:subject], list_lengths[subject+1:]])
            except Exception as e:
                raise ValueError(f"Subject {subject} not found in the data.")
        else:
            try:
                lengths_sum = np.cumsum(list_lengths)
                list_lengths = list_lengths[subject]
                if subject == 0:
                    XX = XX[:list_lengths]
                    YY = YY[:list_lengths]
                else:
                    XX = XX[lengths_sum[subject-1]:lengths_sum[subject]]
                    YY = YY[lengths_sum[subject-1]:lengths_sum[subject]]
            except Exception as e:
                raise ValueError(f"Subject {subject} not found in the data.")

    return XX, YY, list_lengths

def fixation_sequence(self, lengths, method='random', subject=None):
    """
    Generates a sequence of fixation lengths.

    Parameters
    ----------
    lengths : list
        List of fixation lengths for each subject.
    
    method : int or {'random', 'average', 'max'}, default='random
        The method to generate the fixation sequence.
        (based on the original fixation lengths)
        - 'random': Randomly select a fixation length from the list.
        - 'average': Use the average fixation length for all subjects.
        - 'max': Use the maximum fixation length for all subjects.
        - int: Use a specific fixation length for all subjects

    Returns
    -------
    list
        A list of fixation lengths.
    
    Raises
    ------
    ValueError
        If an invalid method is provided.
    """
    if isinstance(method, int):
        if subject is None:
            length = max(lengths)
            total_subjects = method // length
        else:
            length = method
            total_subjects = subject
        return [length]*total_subjects
    else:
        if method == 'random':
            return np.random.randint(min(lengths), max(lengths), len(lengths))
        elif method == 'average':
            avg = np.ceil(np.mean(lengths)).__int__()
            return [avg]*len(lengths)
        elif method == 'max':
            return [max(lengths)]*len(lengths)
        else:
            raise ValueError('Invalid method. Use "random", "average" or "max".')

def data_simulation(self, models, lengths):
    """
    Simulates data from fitted models.

    Parameters
    ----------
    models : list
        A list of fitted models.

    lengths : list
        A list of lengths for each model.

    Returns
    -------
    simulated_data : list
        A list of simulated data for each model.

    simulated_X : list
        A list of simulated X coordinates for each model.

    simulated_Y : list
        A list of simulated Y coordinates for each model.
    """
    simulated_data = []
    for model in models:
        simulations = np.concatenate([model.sample(length)[0] for length in lengths], axis=0)
        simulated_data.append(simulations)
    simulated_X = [data[:, 0].reshape(-1, 1) for data in simulated_data]
    simulated_Y = [data[:, 1].reshape(-1, 1) for data in simulated_data]
    
    return simulated_data, simulated_X, simulated_Y

def models_fit(self, X, Y, models, lengths):
    """
    Fits GaussianHMM models with different numbers of components.

    Parameters
    ----------
    X : np.array
        Input data for the X axis.
    
    Y : np.array
        Input data for the Y axis.
        
    models : list
        List of GaussianHMM models to fit.
        
    lengths : list
        List of lengths for each sequence in X and Y.

    Returns
    -------
    list
        List of fitted GaussianHMM models.
    """
    new_models = []
    for i, model in enumerate(models):
        if X.shape[0] == len(models) :
            fitted = model.fit(np.concatenate((X[i], Y[i]), axis=1), lengths)
        else:
            fitted = model.fit(np.column_stack((X,Y)), lengths)
        new_models.append(fitted)
    return new_models

def model_def(self, X, Y, list_lengths, n_components_list, covariance_type='full', n_iter=10):
    """
    Fits GaussianHMM models with different numbers of components.

    Parameters
    ----------
    X : np.array
        Input data for the X axis.
    
    Y : np.array
        Input data for the Y axis.

    list_lengths : list of int
        The lengths of the sequences in X.

    n_components_list : list of int
        The number of components for each model.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}, default='full'
        The type of covariance parameters for each state.

    n_iter : int, default=10
        The number of iterations to fit the models.
        (correspond to n_iter of `hmmlearn.hmm.GaussianHMM`)

    Returns
    -------
    models : list of `hmmlearn.hmm.GaussianHMM` objects
        The fitted models.

    """
    models = []
    for n_components in n_components_list:
        models.append(hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter))
    models = models_fit(self, X, Y, models, list_lengths)

    return models

def fishers_score_regularized(self, lda, X_train, y_train):
    """
    Calculate the Fisher's Linear Discriminant Score.
    Measures the separation between classes in the transformed feature space. 
    Higher scores indicate better separation, which means HMM model are better able to discriminate between different classes.

    Parameters
    ----------
    lda : LinearDiscriminantAnalysis

    X_train : np.array
        Data simulated from the fitted model.

    y_train : np.array
        Predictions on the simulated data.

    Returns
    -------
    float
        The Fisher's Linear Discriminant Score.
    """
    # Predicting on the simulated data using the fitted model
    X_lda = lda.transform(X_train)

    # Calculate within-class scatter (or within-class covariance) matrix
    # This is the average of the covariance matrices of each class
    # Measures the variability of data points within individual classes
    # Sw = Σ(i=1 to C) Σ(x in Ci) (x - μi)(x - μi)^T
    within_class_scatter = np.zeros((X_train.shape[1], X_train.shape[1]))
    for label in np.unique(y_train):
        class_data = X_lda[y_train == label]
        class_mean = np.mean(class_data, axis=0)
        class_scatter = np.zeros((X_train.shape[1], X_train.shape[1]))
        for sample in class_data:
            sample_diff = (sample - class_mean).reshape(-1, 1)
            class_scatter += np.dot(sample_diff, sample_diff.T)
        within_class_scatter += class_scatter

    # Calculate between-class scatter matrix
    # This measures the variability of data points between different classes
    # Sb = Σ(i=1 to C) Ni * (μi - μ)(μi - μ)^T
    total_mean = np.mean(X_lda, axis=0)
    between_class_scatter = np.zeros((X_train.shape[1], X_train.shape[1]))
    for label in np.unique(y_train):
        class_data = X_lda[y_train == label]
        class_mean = np.mean(class_data, axis=0)
        n = class_data.shape[0]
        mean_diff = (class_mean - total_mean).reshape(-1, 1)
        between_class_scatter += n * np.dot(mean_diff, mean_diff.T)

    alpha = 1e-6  # small regularization parameter
    within_class_scatter_regularized = within_class_scatter + alpha * np.identity(X_train.shape[1])

    # Calculate the Fisher's Linear Discriminant Score with regularization
    # Sum along the diagonal of the dot product of the inverse of the within-class scatter matrix and the between-class scatter matrix
    # FLD = trace(Σw^-1 Σb)
    fishers_score_regularized = np.trace(np.linalg.inv(within_class_scatter_regularized).dot(between_class_scatter))
    return fishers_score_regularized

def score_calculation(self, method, model, data, lengths, lda_model=None):
    """
    Calculate the evaluation score for a model.

    Parameters
    ----------
    method : {'score', 'bic', 'aic', 'lda'}
        The method to calculate the score.

    model : hmmlearn.hmm.GaussianHMM
        The fitted model.

    data : np.array
        The input data.

    lengths : list
        The lengths of the sequences in X.

    lda_model : LinearDiscriminantAnalysis, default=None
        The Linear Discriminant Analysis model.

    Returns
    -------
    float
        The evaluation score for the model.
    """
    match method: 
        case 'score':
            return np.exp(model.score(data, lengths) / data.shape[0])
        case 'bic':
            return model.bic(data, lengths) / data.shape[0]
        case 'aic':
            return model.aic(data, lengths) / data.shape[0]
        case 'lda':
            return fishers_score_regularized(self, lda_model, data, model.predict(data, lengths))
        case _:
            raise ValueError('Invalid method. Use "score", "bic", "aic" or "lda".')

def calculate_starting_likelihood(self, X, Y, list_lengths, n_components_list, covariance_type, n_iter, starting_tests=1, only_bic=False):
    """
    Calculate the likelihood scores for the models with different numbers of components.
    The models are fitted with the real data and the likelihood scores are calculated.

    Parameters
    ----------
    X : np.array
        Input data for the X axis.

    Y : np.array
        Input data for the Y axis.

    list_lengths : list of int
        The lengths of the sequences in X.

    n_components_list : list of int
        The number of components for each model.

    covariance_type : {'full', 'tied', 'diag', 'spherical'}
        The type of covariance parameters for each state.

    n_iter : int, default=10
        The number of iterations to fit the models.
        (correspond to n_iter of `hmmlearn.hmm.GaussianHMM`)

    starting_tests : int, default=10
        The number of starting tests to perform.

    only_bic : bool, default=False
        If `True` only the BIC score is calculated.

    Returns
    -------
    list
        The fitted models.

    np.array
        The log likelihood scores for each model.
        Or `None` if `only_bic` is `True`.

    np.array
        The BIC scores for each model.

    np.array
        The AIC scores for each model.
        Or `None` if `only_bic` is `True`.
    """
    bic_scores_all = []
    if not only_bic:
        log_likelihood_scores_all = []
        aic_scores_all = []

    new_models = []
    best_score = []
    for _ in range(starting_tests):
        # We create every time new models (at the end we will use the last models for the simulation and fitting process)
        models = model_def(self, X, Y, list_lengths, n_components_list, covariance_type, n_iter)
        bic_scores = [score_calculation(self, 'bic', model, np.column_stack((X,Y)), list_lengths) for model in models]
        bic_scores_all.append(bic_scores)
        if not only_bic:
            log_likelihood_scores = [score_calculation(self, 'score', model, np.column_stack((X,Y)), list_lengths) for model in models]
            log_likelihood_scores_all.append(log_likelihood_scores)
            aic_scores = [score_calculation(self, 'aic', model, np.column_stack((X,Y)), list_lengths) for model in models]
            aic_scores_all.append(aic_scores)

        if new_models == []:
            new_models = models
            best_score = bic_scores
        else:
            for i, model in enumerate(models):
                if bic_scores[i] < best_score[i]:
                    new_models[i] = model
                    best_score[i] = bic_scores[i]

    # Convert lists to arrays for easier manipulation
    bic_scores_all = np.array(bic_scores_all)
    if not only_bic:
        log_likelihood_scores_all = np.array(log_likelihood_scores_all)
        aic_scores_all = np.array(aic_scores_all)
    else:
        log_likelihood_scores_all = None
        aic_scores_all = None
    
    return new_models, log_likelihood_scores_all, bic_scores_all, aic_scores_all

def min_max_avg(self, total):
    """
    Calculate the minimum, maximum, and average values for each point in the likelihood matrix.
    
    Parameters
    ----------
    total : np.array
        The likelihood matrix.

    Returns
    -------
    np.array
        The minimum values for each point in the likelihood matrix.

    np.array
        The maximum values for each point in the likelihood matrix.

    np.array
        The average values for each point in the likelihood matrix.
    """
    _min = np.min(total, axis=0)
    _max = np.max(total, axis=0)
    _mean = np.mean(total, axis=0)

    return _min, _max, _mean

def normalize_scores(self, scores):
    """
    Simple min-max normalization of the likelihood matrix.

    Transform the scores matrix to a range between 0 and 1.

    Parameters
    ----------
    scores : np.array
        The likelihood matrix.

    Returns
    -------
    np.array
        The normalized likelihood matrix.
    """
    return (scores - scores.min()) / (scores.max() - scores.min())

def best_models(self, n_components_list, score, threshold):
    """
    Select the top models based on the threshold.

    Parameters
    ----------
    n_components_list : list
        List of numbers of components to fit.

    score : np.array
        The scores for each model.

    threshold : float
        The threshold to select the top models.

    Returns
    -------
    list
        The ranked list of number of components based on the scores.
    """
    sorted_idx = np.argsort(score)
    top_n_models = int(len(n_components_list) * threshold)
    top_models_idx = sorted_idx[:top_n_models]

    # Get ranked models and number of components
    n_components_list_ranked = [n_components_list[i] for i in top_models_idx]

    return n_components_list_ranked

def calculate_best_model(self, n_components_list, bic_scores_all, threshold):
    """
    Calculate the best models based on the mean and standard deviation of the BIC scores.

    Parameters
    ----------
    n_components_list : list
        List of numbers of components to fit.

    bic_scores_all : np.array
        The BIC scores for each model.

    threshold : float, default=0.5
        The threshold to select the top models.

    Returns
    -------
    list
        The ranked list of number of components based on the mean and standard deviation of the BIC scores.

    list
        The ranked list of number of components based on the minimum BIC score.

    Raises
    ------
    ValueError
        If the threshold is not between 0 and 1.
    """
    
    if threshold < 0 or threshold > 1:
        raise ValueError('Invalid threshold. Must be between 0 and 1.')

    min_bic, _, mean_bic = min_max_avg(self, bic_scores_all)
    standard_deviation = np.std(bic_scores_all, axis=0)
    
    # Normalize the scores
    min_bic = normalize_scores(self, min_bic)
    mean_bic = normalize_scores(self, mean_bic)
    standard_deviation = normalize_scores(self, standard_deviation)
    std_score = (mean_bic + standard_deviation) / 2
    
    # Identify best models
    best_model_by_bic = best_models(self, n_components_list, min_bic, threshold)
    n_components_list_ranked = best_models(self, n_components_list, std_score, threshold)

    return n_components_list_ranked, best_model_by_bic

def calculate_entropy(self, stim):
    """
    Calculate the entropy of the fixation data.

    Parameters
    ----------
    stim : int
        The stimulus number.

    Returns
    -------
    float
        The entropy of the fixation data.
    """

    FixCountInd = self.FixCountCalc(stim)
    binnedcount = self.BinnedCount(np.nansum(FixCountInd,0),stim)
    entropy,_ = self.Entropy(binnedcount)

    return entropy

def likelihood_matrix(self, models, simulated_data, lengths, n_components_list, evaluation='score', lda_models=None):
    """
    Creates a likelihood matrix based on the specified criteria.

    Parameters
    ----------
    models : list
        A list of fitted GaussianHMM models.
    
    simulated_data : list
        A list of simulated data.

    lengths : list
        List of number of fixations for each subject.

    n_components_list : list
        List of numbers of components to fit.

    evaluation : {'score', 'bic', 'aic', 'lda', 'all'} or list of str, default='score'
        Criteria to evaluate the models.

    lda_models : list, default=None
        List of LinearDiscriminantAnalysis models.

    Returns
    -------
    np.array
        A likelihood matrix.
    """
    methods = ['score', 'bic', 'aic', 'lda']
    lda = None

    if isinstance(evaluation, str): 
        if evaluation == 'all':
            evaluation = ['score', 'bic', 'aic', 'lda']
        elif evaluation not in methods:
            raise ValueError('Invalid criteria. Use "score", "bic", "aic", "lda" or "all".')
        else:
            likelihood_mat = np.zeros((len(n_components_list), len(n_components_list)))
            for i in range(len(n_components_list)):
                for j, model in enumerate(models):
                    if lda_models is not None:
                        lda = lda_models[j]
                    likelihood_mat[j, i] = score_calculation(self, evaluation, model, simulated_data[i], lengths, lda_model=lda)
            return likelihood_mat
        
    likelihood_mat = np.zeros((len(evaluation), len(n_components_list), len(n_components_list)))
    for i in range(len(n_components_list)):
        for j, model in enumerate(models):
            for k, e in enumerate(evaluation):
                if e in methods:
                    if lda_models is not None:
                        lda = lda_models[j]
                    likelihood_mat[k][j, i] = score_calculation(self, e, model, simulated_data[i], lengths, lda_model=lda)
                else:
                    raise ValueError('Invalid criteria. Use "score", "bic", "aic", "lda" or "all".')
    return likelihood_mat

def create_plot(self, ax, matrix, ticks, title, xlabel, ylabel, method):
    """
    Create a plot for the likelihood matrix.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.

    matrix : np.array
        The likelihood matrix.

    ticks : list
        The ticks for the plot.

    title : str
        The title of the plot.

    xlabel : str
        The label for the x-axis.

    ylabel : str
        The label for the y-axis.

    method : str {score, bic, aic, lda}
        The method ued for evaluation.

    """
    case = {
        'score': 'hot',
        'bic': 'hot_r',
        'aic': 'hot_r',
        'lda': 'hot',
    }
    im = ax.imshow(np.flipud(matrix), cmap=case[method], interpolation='nearest')
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(ticks)))
    ax.set_xticklabels(ticks)
    ax.set_yticks(range(len(ticks)))
    ax.set_yticklabels(np.flip(ticks))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_starting_likelihood(self, n_components_list, bic_scores_all, log_likelihood_scores_all=None, aic_scores_all=None):
    """
    Plot the starting likelihood scores.

    Parameters
    ----------
    n_components_list : list
        List of numbers of components to fit.

    bic_scores_all : np.array
        The BIC scores for each model.

    log_likelihood_scores_all : np.array, default=None
        The log likelihood scores for each model.

    aic_scores_all : np.array, default=None
        The AIC scores for each model.
    """
    # Calculate mean, max, and min for each point in n_components_list
    bic_min, bic_max, bic_mean = min_max_avg(self, bic_scores_all)

    # Plotting
    plt.figure(figsize=(10, 6))

    if log_likelihood_scores_all is None:
        # Plot BIC as error bars
        plt.errorbar(n_components_list, bic_mean, yerr=[bic_mean - bic_min, bic_max - bic_mean], fmt='o-', color='blue')
        plt.title('BIC Scores of Models with Different Components using Real Data')
        plt.xlabel('Number of Components')
        plt.ylabel('Score')
    else:
        log_likelihood_min, log_likelihood_max, log_likelihood_mean = min_max_avg(self, log_likelihood_scores_all)
        aic_min, aic_max, aic_mean = min_max_avg(self, aic_scores_all)

        # Plot BIC and AIC as boxplots
        plt.plot(n_components_list, bic_mean, color='blue', marker='o', label='BIC Mean')
        plt.fill_between(n_components_list, bic_min, bic_max, color='blue', alpha=0.3)
        plt.plot(n_components_list, aic_mean, color='green', marker='o', label='AIC Mean')
        plt.fill_between(n_components_list, aic_min, aic_max, color='green', alpha=0.3)
        lines, labels = plt.gca().get_legend_handles_labels()
        plt.ylabel('Information Criterion Score')

        # Create a second y-axis on the right side for log likelihood
        ax2 = plt.gca().twinx()
        ax2.plot(n_components_list, log_likelihood_mean, color='orange', marker='o', label='Log Likelihood Mean')
        ax2.fill_between(n_components_list, log_likelihood_min, log_likelihood_max, color='orange', alpha=0.3)

        plt.title('Likelihood Scores of Models with Different Components using Real Data')
        plt.xlabel('Number of Components')
        plt.ylabel('Log Likelihood Score')

        # Add legends for both axes
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='upper right')


    # Set x-axis ticks to only n_components_list values
    plt.xticks(n_components_list)


    plt.grid(True)
    plt.show()


def plot_comparison_likelihood(self, n_components_list, scores, labels=None):
    """
    Plot the comparison of likelihood scores for multiple sets of results.

    Parameters
    ----------
    n_components_list : list
        List of numbers of components to fit.
        
    scores : list of lists
        A list where each entry is a list of [log_likelihood_scores_all, bic_scores_all, aic_scores_all]
        for a model.

    labels : list of str, default=None
        A list of labels corresponding to the score arrays for the legend.
    """
    plt.figure(figsize=(10, 6))

    # Plot BIC and AIC scores if available
    for idx, score_set in enumerate(scores):
        log_likelihood_scores_all, bic_scores_all, aic_scores_all = score_set
        
        # Plot BIC scores
        if bic_scores_all is not None:
            bic_min, bic_max, bic_mean = min_max_avg(self, bic_scores_all)
            plt.plot(n_components_list, bic_mean, marker='o', label=f'BIC Group {labels[idx]}' if labels else f'BIC Group {idx+1}')
            plt.fill_between(n_components_list, bic_min, bic_max, alpha=0.3)

        # Plot AIC scores
        if aic_scores_all is not None:
            aic_min, aic_max, aic_mean = min_max_avg(self, aic_scores_all)
            plt.plot(n_components_list, aic_mean, marker='o', label=f'AIC Group {labels[idx]}' if labels else f'AIC Group {idx+1}')
            plt.fill_between(n_components_list, aic_min, aic_max, alpha=0.3)

    # Plot Log Likelihood scores on the second y-axis if available
    if any(score_set[0] is not None for score_set in scores):  # Check if any log likelihood data exists
        ax2 = plt.gca().twinx()  # Create a second y-axis
        for idx, score_set in enumerate(scores):
            log_likelihood_scores_all, _, _ = score_set
            if log_likelihood_scores_all is not None:
                log_likelihood_min, log_likelihood_max, log_likelihood_mean = min_max_avg(self, log_likelihood_scores_all)
                ax2.plot(n_components_list, log_likelihood_mean, color='orange', marker='o', label=f'Log Likelihood Group {labels[idx]}' if labels else f'Log Likelihood Group {idx+1}')
                ax2.fill_between(n_components_list, log_likelihood_min, log_likelihood_max, color='orange', alpha=0.3)

        ax2.set_ylabel('Log Likelihood Score')

    # Final plot customizations
    plt.xlabel('Number of Components')
    plt.ylabel('Information Criterion Score')
    plt.title('Comparison of Likelihood Scores of Models with Different Components')
    
    # Set x-axis ticks to only n_components_list values
    plt.xticks(n_components_list)

    # Combine legends from both axes if log likelihood is plotted
    if any(score_set[0] is not None for score_set in scores):
        lines, labels_left = plt.gca().get_legend_handles_labels()
        lines2, labels_right = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels_left + labels_right, loc='upper right')
    else:
        plt.legend(loc='upper right')

    plt.grid(True)
    plt.show()


def plot_likelihood_matrix(self, likelihood_matrix, n_components_list, iteration, evaluation='score'):
    """
    Plot the likelihood matrix.

    Parameters
    ----------
    likelihood_matrix : np.array
        The likelihood matrix.

    n_components_list : list
        List of numbers of components to fit.

    iteration : int
        The number of iterations.

    evaluation : str or list, default='score'

    """
    if isinstance(evaluation, str):
        if evaluation == 'all':
            evaluation = ['score', 'bic', 'aic', 'lda']
        else:
            fig, ax = plt.subplots(figsize=(10, 10))
            create_plot(self, ax, likelihood_matrix, n_components_list, 'Likelihood Matrix ({})'.format(evaluation), 'Simulated data', 'Fitted model', evaluation)
            plt.show()
            return
        
    num_methods = len(evaluation)
    num_rows = (num_methods + 1) // 2  # Calculate number of rows needed
    fig, axs = plt.subplots(num_rows, 2, figsize=(20, 10*num_rows))
    fig.suptitle('Likelihood Matrix after {} iterations'.format(iteration), fontsize=16)
    for i, method in enumerate(evaluation):
        ax = axs.flat[i]
        create_plot(self, ax, likelihood_matrix[i], n_components_list, 'Likelihood Matrix ({})'.format(method), 'Simulated data', 'Fitted model', method)
    if num_methods % 2 != 0:
        fig.delaxes(axs.flat[-1])
    plt.show()

def plot_simulated_data(self, stim, models, simulated_X, simulated_Y, n_components_list, new_list_len):
    """
    Plot the simulated data for the specified stimulus and models.

    Parameters
    ----------
    stim : int
        The stimulus number.

    models : list
        List of fitted GaussianHMM models.

    simulated_X : list
        List of simulated X-axis eye movement data.

    simulated_Y : list
        List of simulated Y-axis eye movement data.

    n_components_list : list
        List of numbers of components to fit.

    new_list_len : list
        List of fixation lengths for each scanpath.
    """
    subjects = len(new_list_len)
    simulated = np.ceil(np.mean(new_list_len)).__int__()
    total = sum(new_list_len)
    fig,ax=plt.subplots(ncols=len(n_components_list)+1,figsize=(20,5))
    fig.suptitle('Stimulus: {} \n Subject Simulated: {} | Data Simulated for Subject: {} | Total Data Simulated: {}'.format(stim,subjects,simulated,total), fontsize=16)
    self.Heatmap(stim,Vis=1,ax=ax[0],SD=25)
    ax[0].set_title('Oringinal Heatmap')
    for i, (n_components, simulated_X, simulated_Y, model) in enumerate(zip(n_components_list, simulated_X, simulated_Y, models)):
        self.VisHMM(np.column_stack((simulated_X, simulated_Y)),model,ax=ax[i+1],stim=stim,lengths=new_list_len)
        ax[i+1].set_title('Number of components: {}'.format(n_components))
    plt.show()

def plot_models_summary(self, likelihood_matrix, n_components_list, evaluation):
    """
    Plot a summary of all the models based on the likelihood matrix and different evaluation criteria.
    For each model, the diagonal value is checked and if is well separable the best result is highlighted in red.

    Parameters
    ----------
    likelihood_matrix : list
        A list of likelihood matrices.
    
    n_components_list : list
        List of numbers of components to fit.

    evaluation : str or list
        Criteria to evaluate the models.
    """
    final_scores = np.zeros((len(n_components_list), len(n_components_list)))
    if isinstance(evaluation, str):
        # If evaluation is 'all' the final score is the average of all our scores matrix
        if evaluation == 'all':
            for index, lm in enumerate(likelihood_matrix):
                # normalize the likelihood matrix
                lm = normalize_scores(self, lm)
                # add the normalized likelihood matrix to the final scores
                if index == 1 or index == 3:
                    final_scores += lm
                else:
                    final_scores -= lm
            final_scores = final_scores / len(likelihood_matrix)
        else:
            final_scores = (likelihood_matrix - likelihood_matrix.min()) / (likelihood_matrix.max() - likelihood_matrix.min())
    else:
        for index, ev in enumerate(evaluation):
            lm = likelihood_matrix[index]
            # normalize the likelihood matrix
            lm = normalize_scores(self, lm)
            # add the normalized likelihood matrix to the final scores
            if ev == 'score' or ev == 'lda':
                final_scores += lm
            else:
                final_scores -= lm
        final_scores = final_scores / len(evaluation)

    _, ax = plt.subplots(figsize=(10, 10))
    create_plot(self, ax, final_scores, n_components_list, 'Final Scores Matrix', 'Simulated Data', 'Fitted Model', 'score')
    # plot a diagonal line from the bottom-left to the top-right
    ax.plot([0, len(n_components_list)-1], [len(n_components_list)-1, 0], color='gray', linestyle='--')

    # for every row check the diagonal value from the bottom-left to the top-right and print the best model
    x = len(n_components_list)-1
    y = 0
    for i in range(len(n_components_list)-1, -1, -1):    
        col = [row[x] for row in final_scores]
        row = final_scores[i]
        max_val = np.max(col)
        if final_scores[i][x] == max_val:
            row = np.delete(row, i)
            col.pop(i)
            col_val = max_val - np.mean(col) - np.mean(row)
            ax.text(i, y, round(col_val, 2), ha='center', va='center', color='green', fontsize=12)
        x -= 1
        y += 1
    plt.show()

def plot_pipeline(self, models, stimuli, iteration, simulated_X, simulated_Y, n_components_list, new_list_len, likelihood_mat, evaluation, Summary=True):
    """
    Plot the results of the pipeline.

    If `Summary` is `False` only the simulated data and the likelihood matrix (if evaluation is a list or 'all') are plotted.

    Parameters
    ----------
    models : list
        List of GaussianHMM models.

    stimuli : int or list
        The stimulus number or a list of stimulus numbers.

    iteration : int
        Number of iterations for simulating and fitting the models.

    simulated_X : list
        List of simulated X-axis eye movement data.

    simulated_Y : list
        List of simulated Y-axis eye movement data.

    n_components_list : list
        List of numbers of components to fit.

    new_list_len : list
        List of fixation lengths for each scanpath.

    likelihood_mat : np.array
        The likelihood matrix.

    evaluation : str
        Criteria to evaluate the models.

    Summary : bool, default=True
        Print the best model based of the likelihood matrix.
        - If `False` the complete likelihood matrix is plotted.
    """
    if not Summary:
        plot_likelihood_matrix(self, likelihood_mat, n_components_list, iteration, evaluation)
    plot_simulated_data(self, stimuli, models, simulated_X, simulated_Y, n_components_list, new_list_len)
    if len(n_components_list) > 1:
            if evaluation == 'all' or isinstance(evaluation, list):
                plot_models_summary(self, likelihood_mat, n_components_list, evaluation)


def models_pipeline(self, stim, n_components_list, group=-1, select_subj=-1, remove_subj=False, group_subj=-1, iteration=1, tollerance=20, simulation_type='random', evaluation='all', covariance_type='full', n_iter=10, starting_tests=1, only_starting=False, only_bic=False, only_best=False, threshold=0.5, list_models=None, subject=None):
    """
    Initializes the GaussianHMM models and simulates data for the specified stimulus, then fits the models and evaluates them.
    Repeat the simulation and fitting process for the specified number of iterations.
    Returns the models, simulated data (with the list of fixation for every scanpath) and likelihood matrix.

    If only the starting likelihood is needed, set `only_starting` to `True`.
    If the user wants to proceed with the evaluation selecting automatically the best models based on the concordance, set `concordance` to `True` (also the threshold can be set to select the top models).

    The resulting parameters are compatible with the `plot_pipeline` function.
    
    Parameters
    ----------
    stim : int
        The stimulus number.

    n_components_list : list
        List of the number of components for each subject.

    covariance_type : str
        Type of covariance for the `hmmlearn.hmm.GaussianHMM` models.

    iteration : int
        Number of iterations for simulating and fitting the models.

    evaluation : str
        Criteria to evaluate the models.

    stimulation_type : str
        Type of stimulation for the fixation sequence.
    
    n_iter : int
        Number of iterations for fitting the models.

    starting_tests : int
        Number of starting tests to perform.

    only_starting : bool
        If `True` only the starting likelihood is calculated.

    only_bic : bool
        If `True` only the BIC score is calculated.

    only_best : bool
        If `True` the best model is selected based on the threshold.

    threshold : float
        The threshold to select the top models.

    list_models : list
        List of GaussianHMM models to fit.
        Can be used to start the pipeline from a specfic set of pre-fitted models.

    Returns
    -------
    models : list
        List of GaussianHMM models fitted.

    simulated_data : np.array
        Simulated eye movement data.

    simulated_X : np.array
        Simulated X-axis eye movement data.

    simulated_Y : np.array
        Simulated Y-axis eye movement data.

    list_len : list
        List of number of components for each model
    """

    if list_models is not None:
        models = copy.deepcopy(list_models)
    else:
        models = None

    list_results = []
    scores = []

    if isinstance(group, int):
        group = [group]

    original_components = n_components_list

    for g in group:
        if group_subj != -1:
            if g == group_subj:
                X, Y, list_lengths = get_data(self, stim, g, tollerance, select_subj, remove_subj)
            else:
                X, Y, list_lengths = get_data(self, stim, g, tollerance)
        else:
            X, Y, list_lengths = get_data(self, stim, g, tollerance, select_subj, remove_subj)

        if starting_tests < 1:
            raise ValueError('Invalid number of starting tests. Must be greater or equal to 1.')
        
        if models is None:
            # Calculate the likelihood scores for the models
            models, log_likelihood_scores_all, bic_scores_all, aic_scores_all = calculate_starting_likelihood(self, X, Y, list_lengths, n_components_list, covariance_type, n_iter, starting_tests, only_bic)
            # Send the results to the plot function
            if len(group) == 1:
                plot_starting_likelihood(self, n_components_list, bic_scores_all, log_likelihood_scores_all, aic_scores_all)
            if len(group) > 1:
                scores.append([log_likelihood_scores_all, bic_scores_all, aic_scores_all])

            results = None
            if only_best:
                n_components_list, best_bic = calculate_best_model(self, n_components_list, bic_scores_all, threshold)
                entropy = calculate_entropy(self, stim)
                # create a dataframe with the results
                results = pd.DataFrame({
                    'Stimulus': [stim],
                    'Best Model (Min BIC)': [best_bic],
                    'Best Model (Mean + Standard Deviation)': [n_components_list],
                    'Entropy': [entropy]
                })
                print(results)

                # Reorder the models based on the n components list
                n_components_list = np.sort(n_components_list)
                new_models = []
                for model in models:
                    i = model.n_components
                    if i in n_components_list:
                        new_models.append(model)
                models = new_models

            # If only the starting likelihood is needed, return the models
            # Is possible to use this function to get the starting likelihood and then use the models for other purposes
            if only_starting:
                list_results.append([models, results])
                models, results = None, None
                continue
        else:
            if len(models) != len(n_components_list):
                n_components_list = [model.n_components for model in models]

        lda_models = None
        if (isinstance(evaluation, list) and 'lda' in evaluation) or (isinstance(evaluation, str) and evaluation in {'lda', 'all'}):
            lda_models = []
            for model in models:
                data = np.column_stack((X,Y))
                y_train = model.predict(data)
                lda = LinearDiscriminantAnalysis()
                lda.fit(data, y_train)
                lda_models.append(lda)

        list_len =  fixation_sequence(self, list_lengths, simulation_type, subject)
        # Simulate data from the models
        simulated_data, simulated_X, simulated_Y = data_simulation(self, models, list_len)

        if iteration > 1:
            new_models = copy.deepcopy(models)
            sd, sx, sy = simulated_data, simulated_X, simulated_Y
            new_list_len = list_len
            # Evaluate the models and repete the simulation and fitting process for the specified number of iterations
            likelihood_mat = likelihood_matrix(self, new_models, sd, new_list_len, n_components_list, evaluation, lda_models=lda_models)
            for _ in range(iteration-1):
                # The fit function is printing that the parameters ‘s’ for startprob, ‘t’ for transmat, ‘m’ for means, and ‘c’ for covars 
                # are changing, we do not want to see this
                with io.capture_output() as _:
                    new_models = models_fit(self, np.array(sx), np.array(sy), new_models, new_list_len)
                # Simulate data from the models (fitted with real data)
                new_list_len = fixation_sequence(self, list_lengths, simulation_type, subject)
                sd, sx, sy = data_simulation(self, models, new_list_len) 
                likelihood_mat += likelihood_matrix(self, new_models, sd, new_list_len, n_components_list, evaluation, lda_models=lda_models)
            models = new_models
            simulated_X = sx
            simulated_Y = sy
            list_len = new_list_len
        else:
            # Evaluate the models
            likelihood_mat = likelihood_matrix(self, models, simulated_data, list_len, n_components_list, evaluation, lda_models=lda_models)
        
        list_results.append([models, simulated_X, simulated_Y, list_len, likelihood_mat, n_components_list])
        models, simulated_X, simulated_Y, list_len, likelihood_mat = None, None, None, None, None
        n_components_list = original_components

    if len(group) > 1:
        plot_comparison_likelihood(self, n_components_list, scores, labels = group)

    return list_results

def GaussianHMMPipeline(self, stim, n_components_list, group=-1, iteration=1, tollerance=20, simulation_type='random', evaluation='all', covariance_type='full', n_iter=10, Summary=True, starting_tests=1, only_starting=False, only_bic=False, only_best=False, threshold=0.5, list_models=None, subject=None):
    """
    Pipeline for helping the user in create, fit and evaluate GaussianHMM models with different numbers of components.
    
    This function takes a stimulus or a list of stimulus, retrive the fixatian data, and fits GaussianHMM models with different numbers of components.
    Then it simulates data end evaluate the models, also it can fit again the models and repete all the process for the specified number of iterations.
    The `Summary` parameter is used to choose what plot, set to `False` if interested only the simulated data and the likelihood matrix (if evaluation is a list or 'all') are plotted.
    If the user wants to see only the starting likelihood scores, it can be done setting `only_starting` to `True`.
    If the user wants to proceed with the evaluation selecting automatically the best models based it can be done setting `concordance` to `True` (also the threshold can be set to select the top models).

    The evaluation can be done using different criteria:
    - `'score'`: the log likelihood of the data under the model, which is a measure of how well the model explains the data.
    - `'bic'`: Bayesian Information Criterion, which is a trade-off between the likelihood of the data and the number of parameters in the model.
    - `'aic'`: Akaike Information Criterion, which is similar to BIC but with a different penalty term.
    - `'lda'`: Fisher's Linear Discriminant Score, which measures the separation between classes in the transformed feature space.
    - `'all'`: all the above criteria.


    Then all the results are plotted to help the user to understand how the models are performing.

    Parameters
    ----------
    stim: (int, list) 
        The stimulus number or a list of stimulus numbers.
    
    n_components_list: (list)
        List of numbers of components to fit.
    
    evaluation : (str, optional)
        Criteria to evaluate the models ('score', 'bic', 'aic', 'lda' or 'all') or list of str.
        Defaults to 'all'.
    
    covariance_type : {'full', 'diag', 'spherical', 'tied'}, default: 'full'
        Covariance type for the `hmmlearn.hmm.GaussianHMM`.
    
    iteration : int, default: 1
        Number of `data_simulation` and `models_fit` iterations.
    
    Summary : bool, default: True
        Print the best model based of the likelihood matrix.
        - If `False` the complete likelihood matrix is plotted.

    simulation_type : {'random', 'average', 'max'}, default: 'random'
        The method to generate the fixation sequence, used in `fixation_sequence`.

    starting_tests : int, default: 10

    only_starting : bool, default: False
        If `True` the function will only plots the models after the first fitting.

    only_bic : bool, default: False
        If `True` only the BIC score is calculated.

    only_best : bool, default: False
        If `True` the best model is selected based on the threshold.

    threshold : float, default: 0.5
        The threshold to select the top models.

    models : list, default: None
        List of GaussianHMM models to fit.
        Can be used to start the pipeline from a specfic set of pre-fitted models.
        
    Raises
    ------
        ValueError: If the parameters are invalid.
        ValueError: If the intialization of the models fails.

    Examples
    --------
        >>> Data = PyEyeSim.EyeData(...)
        >>> Data.DataInfo(FixDuration='length',Stimulus='filenumber',subjectID='SUBJECTINDEX',StimPath=...,StimExt='.png')
        >>> Data.RunDescriptiveFix()
        >>> stimuli = (Data.stimuli).tolist()
        >>> GaussianHMMPipeline(stimuli, [2, 3, 5, 7])
    """
    if list_models is not None:
        models = copy.deepcopy(list_models)
    else:
        models = None

    if iteration < 1:
        raise ValueError('Invalid number of iterations. Must be greater or equal to 1.')

    if isinstance(evaluation, list) and len(evaluation) == 1:
        evaluation = evaluation[0]

    if isinstance(group, int):
        group = [group]

    for g in group:
        if isinstance(stim, int) or isinstance(stim, float): 
            try:
                result = models_pipeline(self, stim, n_components_list, g, iteration, tollerance, simulation_type, evaluation, covariance_type, n_iter, starting_tests, only_starting, only_bic, only_best, threshold, models, subject)
                if only_starting:
                    return result
                else:
                    models, simulated_X, simulated_Y, new_list_len, likelihood_mat, n_components_list = result[0]
            except Exception as e:
                print("Stimulus: {} - Error: {}".format(stim, e))
                print('Check the parameters and try again.')
                return 
            # Normalize the likelihood matrix for the number of iterations
            likelihood_mat = likelihood_mat/iteration
            plot_pipeline(self, models, stim, iteration, simulated_X, simulated_Y, n_components_list, new_list_len, likelihood_mat, evaluation, Summary)
        elif isinstance(stim, list):
            results = []
            for s in stim:
                try:
                    result = models_pipeline(self, s, n_components_list, g, iteration, tollerance, simulation_type, evaluation, covariance_type, n_iter, starting_tests, only_starting, only_bic, only_best, threshold, models, subject)
                    if only_starting:
                        results.append(result)
                        continue
                    else:
                        models, simulated_X, simulated_Y, new_list_len, likelihood_mat, new_components_list = result[0]
                except Exception as e:
                    print("Stimulus: {} - Error: {}".format(stim, e))
                    print('Check the parameters and try again.')
                    print('Skipping to the next stimulus...')
                    continue
                # Normalize the likelihood matrix for the number of iterations 
                for i in range(len(likelihood_mat)):
                    likelihood_mat[i] = likelihood_mat[i]/iteration
                plot_pipeline(self, models, s, iteration, simulated_X, simulated_Y, new_components_list, new_list_len, likelihood_mat, evaluation, Summary)
            if only_starting:
                return results
        else:
            raise ValueError('Invalid stimulus input. Use an integer or a list of integers.')