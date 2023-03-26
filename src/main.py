# Basic libraries
import pandas as pd
import numpy as np
# Modeling and evaluation librarier
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Library for the CLI
import argparse
import sys




# Setting up the arguments for the CLI
parser = argparse.ArgumentParser(description="Regression model training")

parser.add_argument('-fp', '--filepath', type=str,
                    help='Type in the filepath of the .csv file with your dataset. Default is \'dataset_dir\winequality.csv\' ',
                    default='dataset_dir/winequality.csv')

parser.add_argument('-l', '--label', type=str,
                    help='name of the column that contrains the label (the y, or the value we are predicting). Default is quality',
                    default='quality')

parser.add_argument('-tf', '--train_features', type=str, nargs='*',
                    help='Features (columns of csv) for the model to make the predictions. Default uses all. Write your columns names afer each other. e.g. -mf pH sulphates. If there the feature consists of multiple words write it in double quotes like this: \"fixed acidity\". If the number of feature is one or two it will create visualizations for those (2D if only one feature is present, 3D if two features are present)')

parser.add_argument('-f', '--features', type=bool,
                    help='print out the features. Default is False', choices=[True, False],
                    default=False)

parser.add_argument('-s', '--split', type=float,
                    help='The percentage of the train-test split (the value you pass will give you the train percentage). Default is: 0.25',
                    default=0.25)

parser.add_argument('-op', '--output_path', type=str,
                    help='Specify the path of the output file. It will contain the evaluation metrics, and the predicted values with true values in a separate file. Default is the currect dir',
                    default='')

parser.add_argument('-r', '--round', type=bool,
                    help='Round the results (True or False). If True will automatically create a confusion matrix plot and calculate the accuracy of the model. Default is False',
                    default=False)

parser.add_argument('-sk', '--svm_kernel', type=str,
                    help='The typeof the kernel for the SVR model. Default is rbf', choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    default='rbf')

parser.add_argument('-de', '--degree', type=int,
                    help='The degree of the polynomial if the kernel is \'poly\' (ignored by other kernels). Default is 3.',
                    default=3)

parser.add_argument('-ga', '--gamma', type=str,
                    help='Kernel coefficient for \'rbf\', \'poly\' and \'sigmoid\'. Default is scale',
                    choices=['scale', 'auto'],
                    default='scale')

parser.add_argument('-co', '--coef0', type=float,
                    help='Independent term in kernel function. It is only significant in \'poly\' and \'sigmoid\'. Default is 0.0',
                    default=0.0)

parser.add_argument('-to', '--tol', type=float,
                    help='Tolerance for stopping criterion. Default is 1e-3',
                    default=1e-3)

parser.add_argument('-c', '--C', type=float,
                    help='Regularization parameter The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty. Default is 1.0',
                    default=1.0)

parser.add_argument('-e', '--epsilon', type=float,
                    help='Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value. Must be non-negative Default is 0.1.',
                    default=0.1)

parser.add_argument('-sh', '--shrinking', type=bool,
                    help='Whether to use the shrinking heuristic. See the User Guide. Default is True',
                    default=True)

parser.add_argument('-cs', '--cache_size', type=float,
                    help='Specify the size of the kernel cache (in MB). Default is 200.0',
                    default=200.0)

parser.add_argument('-v', '--verbose', type=bool,
                    help='Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context. Default is False',
                    default=False)

parser.add_argument('-mi', '--max_iter', type=int,
                    help='Hard limit on iterations within solver, or -1 for no limit. Default is -1',
                    default=-1)
parser.add_argument("-hr", "--html_report", type=bool,
                    help="Save the figures and the tables to a html file so you can view it in a web browser in an interactive form. Default is False",
                    default=False)
args = parser.parse_args()

# Reading the datafram with pandas
df = pd.read_csv(args.filepath)
if args.features:  # print out the features of the dataset if the user asked for it
    print('The features are: ', df.columns.to_list())
    print('\n')
    sys.exit()
X = df.drop(columns=args.label)

# If the user didn't specify the features it will automatically use every feature. If they specified it than it will use those
if args.train_features == None:
    pass
else:
    X = X[args.train_features]
y = df[args.label]

# splitting the dataset for train-test sets where the user can specify the % of the split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.split, random_state=42)

# Defining the model where every parameter can be given by the user
model = SVR(kernel=args.svm_kernel, degree=args.degree, gamma=args.gamma, coef0=args.coef0,
            tol=args.tol, C=args.C, epsilon=args.epsilon, shrinking=args.shrinking,
            cache_size=args.cache_size, verbose=args.verbose, max_iter=args.max_iter)

# fitting the model for the training data
model.fit(X_train, y_train)
# Creating predictions with the data
predictions = model.predict(X_test)
# printing out the predictions
print('predicted values are:', predictions)

# Creating a 2D plot if the user only specified one feature. This way we can evaluate the result based on the plot
if args.train_features != None and len(args.train_features) == 1:
    x = X_test
    y = y_test
    fig, ax= plt.subplots()
    ax.scatter(x, y, label='true_values')
    ax.scatter(x, predictions, color='r', label='predicted')
    ax.set_title('Simple regression with the given model')
    ax.set_xlabel(args.train_features[0])
    ax.set_ylabel(args.label)
    ax.legend()
    plt.show()
    fig.savefig(args.output_path+"1_feature_regression.png")

    # Creates a figure for the HTML report
    if args.html_report==True:
        fig_line=go.Figure(make_subplots(rows=1, cols=1))
        fig_line.add_trace(go.Scatter(x=X_test.values.ravel(),y=y_test.values.ravel(),  mode='markers', name="true"), row=1, col=1)
        fig_line.add_trace(go.Scatter(x=X_test.values.ravel(),y=predictions,  mode='markers', name="predicted"), row=1, col=1)
        fig_line.update_xaxes(title_text=args.train_features[0], row=1, col=1)
        fig_line.update_yaxes(title_text=args.label, row=1, col=1)
    else:
        pass
# If the user specified two features the program will create a 3D visualization so we can assess the results. args.train_features!=None is required because
# if there are no features given it will be a None value and it would give an error
elif args.train_features != None and len(args.train_features) == 2:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x = X_test[args.train_features[0]]
    y = X_test[args.train_features[1]]
    z = y_test
    ax.scatter3D(x, y, z, c=z)
    ax.plot_trisurf(x, y, predictions, linewidth=0.2, antialiased=False)
    ax.set_xlabel(args.train_features[0])
    ax.set_ylabel(args.train_features[1])
    ax.set_zlabel(args.label)
    plt.show()
    fig.savefig(args.output_path+"2_feature_regression.png")

    # Creates a figure for the HTML report
    if args.html_report==True:
        fig_3D=go.Figure(make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]]))
        fig_3D.add_trace(go.Scatter3d(x=x.values.ravel(), y=y.values.ravel(), z=z.values.ravel(), mode='markers', name='True'), row=1, col=1)
        fig_3D.add_trace(go.Scatter3d(x=x.values.ravel(), y=y.values.ravel(), z=predictions, mode='markers', name='Predicted'), row=1, col=1)
        fig_3D.update_layout(scene=dict(xaxis_title=args.train_features[0],
                                        yaxis_title=args.train_features[1],
                                        zaxis_title=args.label))

    else:
        pass

else:
    pass

# Since this task is close to a classification task I added an option to round the output to make it more interpretable.
# This will also crate a confusion matrix so we can also visualize those results.
if args.round==True:
    rounded_pred = predictions.round()
    acc = accuracy_score(y_test, rounded_pred)

    # Here we can save the models output to a .csv file that stores the different metrics used for regression analysis
    # (MSE, RMSE, MAE, R^2) and in this case if the round is True than it will save the accuracy of the model and save the rounded values to the predictions
    # it saves the results without the accuracy column.
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    svr_score = model.score(X_test, y_test)
    eval_metrics = pd.DataFrame({'Mean_Squared_Error': mse,
                                 'Root_Mean_Squared_Error': mse**0.5,
                                'Mean_Absolute_Error': mae,
                                 'SRV_coefficicient': svr_score,
                                 'accuracy_rounded': acc}, index=[0])
    eval_values = pd.DataFrame({'true_values': y_test,
                               'rounded_predictions': rounded_pred,
                              'predicted_values': predictions})

    eval_metrics.to_csv(args.output_path +
                        'evaluation_metrics.csv', index=False)
    eval_values.to_csv(args.output_path + 'evaluation_values.csv', index=False)
    
    # Here on top of the confusion matrix we create a
    # residuals plot, histogram of residuals, a plot where I plotted the predicted values against the y values.
    # It creates these to assess the regression models performance
    residuals = y_test-predictions
    fig, axs = plt.subplots(2, 2, figsize=(13, 8))
    sns.heatmap(confusion_matrix(y_test, rounded_pred),
                annot=True, ax=axs[0, 0])
    axs[0, 0].set_title('Confusion Matrix')
    axs[0, 0].set_ylabel('Actal Values')
    axs[0, 0].set_xlabel('Predicted Values')
    axs[0, 1].scatter(x=y_test, y=residuals)
    axs[0, 1].axhline(y=0, color='red', linestyle='-')
    axs[0, 1].set_title('Residual plot')
    axs[0, 1].set_xlabel('The actual values')
    axs[0, 1].set_ylabel('residuals values')
    axs[1, 0].scatter(x=y_test, y=predictions)
    axs[1, 0].set_title('predicted values x real values')
    axs[1, 0].set_xlabel('The actual values')
    axs[1, 0].set_ylabel('predicted values values')
    axs[1, 1].hist(residuals)
    axs[1, 1].set_title('Residual histogram')
    axs[1, 1].set_ylabel('residuals values')
    fig.subplots_adjust(hspace=.5)
    plt.show()
    fig.savefig(args.output_path+"evalauation_figure.png", dpi=300)

    #Generating the heatmap and the rounded values for the tables
    if args.html_report==True:
        #Defining the values for the HTML report
        eval_met_list=[eval_metrics.Mean_Squared_Error,
                eval_metrics.Root_Mean_Squared_Error,
                eval_metrics.Mean_Absolute_Error,
                eval_metrics.SRV_coefficicient,
                eval_metrics.accuracy_rounded]
        
        eval_val_list=[eval_values.true_values,
                       eval_values.rounded_predictions,
                       eval_values.predicted_values]
        
        fig_heat=go.Figure(make_subplots(rows=1, cols=1, specs=[[{"type":"heatmap"}]],
                                                        subplot_titles=("confusion_matrix_heatmap",)))
        fig_heat.add_trace(go.Heatmap(z=confusion_matrix(y_test, rounded_pred), 
                                        text=confusion_matrix(y_test, rounded_pred),
                                                                texttemplate="%{text}",
                                                                textfont={"size":20}), row=1, col=1)
        fig_heat.update_xaxes(title_text='Predicted Values', row=1, col=1)
        fig_heat.update_yaxes(title_text='Actal Values', row=1, col=1)
    else:
        pass

# If there the round is not true the confusion matrix is skipped and it just creates a
# residuals plot, histogram of residuals, a plot where I plotted the predicted values against the y values and it saves the results without the accuracy column.
else:
    residuals = y_test-predictions
    fig, axs = plt.subplots(1, 3, figsize=(13, 7))
    axs[0].scatter(x=y_test, y=residuals)
    axs[0].axhline(y=0, color='red', linestyle='-')
    axs[0].set_title('Residual plot')
    axs[0].set_xlabel('The actual values')
    axs[0].set_ylabel('residuals values')
    axs[1].scatter(x=y_test, y=predictions)
    axs[1].set_title('predicted values x real values')
    axs[1].set_xlabel('The actual values')
    axs[1].set_ylabel('predicted values')
    axs[2].hist(residuals)
    axs[2].set_title('Residual histogram')
    axs[2].set_ylabel('residuals values')
    plt.show()
    fig.savefig(args.output_path+"evalauation_figure.png")

    # and here we also save our results.
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    svr_score = model.score(X_test, y_test)
    eval_metrics = pd.DataFrame({'Mean_Squared_Error': mse,
                                 'Root_Mean_Squared_Error': mse**0.5,
                                 'Mean_Absolute_Error': mae,
                                 'SRV_coefficicient': svr_score}, index=[0])
    eval_values = pd.DataFrame({'true_values': y_test,
                              'predicted_values': predictions})

    eval_metrics.to_csv(args.output_path +
                        'evaluation_metrics.csv', index=False)
    eval_values.to_csv(args.output_path + 'evaluation_values.csv', index=False)

    eval_met_list=[eval_metrics.Mean_Squared_Error,
                eval_metrics.Root_Mean_Squared_Error,
                eval_metrics.Mean_Absolute_Error,
                eval_metrics.SRV_coefficicient,]
        
    eval_val_list=[eval_values.true_values,
                eval_values.predicted_values]

# Creates all the figures for the HTML report if the user specified True
if args.html_report==True:
        y_eq_0 = [0 for i in range(0, len(y_test))] # This creates all 0-s for the y=0 line for the residual plot
        with open(args.output_path +'Report.html', 'w+') as f:
            # Initialize the Figure. Also specifying the types in the specs and the titles
            fig_base=go.Figure(make_subplots(rows=5, cols=1, specs=[[{"type": "table"}],
                                                                [{"type":"table"}],
                                                                [{"type": "scatter"}],
                                                                [{"type": "scatter"}],
                                                                [{"type": "histogram"}]], subplot_titles=("Evaluation_table", 
                                                                                                                             "Values_table",
                                                                                                                            "Residual_plot",
                                                                                                                            "Predicted_values_x_real_values",
                                                                                                                            "Residual_Histogram")))
            # Creating the first tabe for the evaluation metrics
            fig_base.add_trace(go.Table(header=dict(values=list(eval_metrics.columns)), cells=dict(values=eval_met_list)), row=1, col=1)
            # Creating the tabe for the evaluation values
            fig_base.add_trace(go.Table(header=dict(values=list(eval_values.columns)), cells=dict(values=eval_val_list)), row=2, col=1)
            

            fig_base.add_trace(go.Scatter(x=y_test, y=residuals, mode='markers', name='residuals'), row=3,col=1)
            fig_base.add_trace(go.Scatter(x=y_test, y=y_eq_0, mode="lines", name="y=0"), row=3, col=1)
            fig_base.update_xaxes(title_text='Actal Values', row=3, col=1)
            fig_base.update_yaxes(title_text='Residuals', row=3, col=1)


            fig_base.add_trace(go.Scatter(x=y_test, y=predictions, mode='markers', name='actual x pred'), row=4, col=1)
            fig_base.update_xaxes(title_text='Actal Values', row=4, col=1)
            fig_base.update_yaxes(title_text='Predicted Values', row=4, col=1)


            fig_base.add_trace(go.Histogram(x=residuals, name='residual_histogram'), row=5, col=1)
            fig_base.update_yaxes(title_text='residual count', row=5, col=1)

            fig_base.update_layout(showlegend=True, height=1500)
            # Saving the base plots to a HTML file
            f.write(fig_base.to_html(full_html=False, include_plotlyjs='cdn'))

            # Write the HTML files if they met the condition
            if args.round==True:
                f.write(fig_heat.to_html(full_html=False, include_plotlyjs='cdn'))

            if args.train_features != None and len(args.train_features) == 1:
                f.write(fig_line.to_html(full_html=False, include_plotlyjs='cdn'))

            if args.train_features != None and len(args.train_features) == 2:
                f.write(fig_3D.to_html(full_html=False, include_plotlyjs='cdn'))
                   
            print("report generated")
else:
    pass