



# Work by MUJTABA MATEEN
# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
import warnings
warnings.simplefilter('ignore')
sns.set_theme(style="dark")



# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
import warnings
warnings.simplefilter('ignore')
sns.set_theme(style="dark")


data = pd.read_csv("./weatherHistory.csv")
data.head()
print(data)

     


print("------------")

data.info()

print()




print(data.describe())


print("----------------------")


print(data["Summary"].value_counts())

data = data[(data["Summary"] == "Overcast") | (data["Summary"] == "Clear") | (data["Summary"] == "Foggy")]
data.info()


data.dropna(inplace=True) 
# Again checking for values
missing_values_count = data.isnull().sum()
missing_values_count


float_cols = data.select_dtypes(include='float')
data[float_cols.columns] = float_cols.round(2)
data.head()


# Formatting Date Column. This can be used to identify any seasonality and trends
data['Formatted Date'] = pd.to_datetime(data['Formatted Date'], errors='coerce')
# Extracting the relevant components
data["Time"] = [d.time() for d in data['Formatted Date']]
data["Time"] = data["Time"].astype(str)
data["Time"] = data["Time"].str.split(':').str[0].astype(int)
data["Date"] = [d.date() for d in data['Formatted Date']]
data["Date"]= data["Date"].astype(str)
data["Year"] = data["Date"].str.split('-').str[0].astype(int)
data["Month"] = data["Date"].str.split('-').str[1].astype(int)
data["Day"] = data["Date"].str.split('-').str[2].astype(int)
# Dropping the original column 
data = data.drop(columns=['Formatted Date','Date'], axis=1)

# It can be seen that the feature "Loud Cover" have only value '0' and mean and other statistical overview also support the deduction. Hence, it is the redundant column
data["Loud Cover"].value_counts()


numeric_columns = list(data.select_dtypes(include=['float64', 'int64']).columns)
categorical_columns = list(data.select_dtypes(include=['object']).columns)
continuous_columns = [i for i in numeric_columns if len(list(data[i].unique()))>=25]
discrete_columns = [i for i in numeric_columns if len(list(data[i].unique()))<25]
print("Numerical Columns: ", numeric_columns)
print()
print("Categorical Columns: ", categorical_columns)
print()
print("Continuous Columns: ", continuous_columns)
print()
print("Discrete Columns: ", discrete_columns)


# plt.figure(figsize=(18, 8)) 
# sns.boxplot(data=data[numeric_columns])
# plt.show()

def remove_outliers(df, feature):
    """
    Remove Outliers using IRQ method
    
    df: dataframe
    feature: dataframe column"""
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr
    df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

# Removing Outliers
data = remove_outliers(data, "Pressure (millibars)")
data = remove_outliers(data, "Wind Speed (km/h)")
data = remove_outliers(data, "Humidity")
data = remove_outliers(data, "Temperature (C)")
data = remove_outliers(data, "Apparent Temperature (C)")

plt.figure(figsize=(18, 8)) 
sns.boxplot(data=data[numeric_columns])
plt.show()



def histogram_boxplot(data, feature, figsize=(15, 10), kde=False, bins=None):
    """
    Boxplot and histogram combined

    data: dataframe
    feature: dataframe column
    figsize: size of figure (default (15,10))
    kde: whether to show the density curve (default False)
    bins: number of bins for histogram (default None)
    """
    f2, (ax_box2, ax_hist2) = plt.subplots(
        nrows=2,  
        sharex=True,  
        gridspec_kw={"height_ratios": (0.25, 0.75)},
        figsize=figsize,
    )  
    sns.boxplot(
        data=data, x=feature, ax=ax_box2, showmeans=True, color="violet"
    )
    sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2, bins=bins
    ) if bins else sns.histplot(
        data=data, x=feature, kde=kde, ax=ax_hist2
    ) 
    ax_hist2.axvline(
        data[feature].mean(), color="green", linestyle="--"
    ) 
    ax_hist2.axvline(
    )  
    plt.tight_layout()

    plt.show()




def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 2, 6))
    else:
        plt.figure(figsize=(n + 2, 6))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        palette="Paired",
        order=data[feature].value_counts().index[:n],
    )

    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  
        else:
            label = p.get_height() 

        x = p.get_x() + p.get_width() / 2  
        y = p.get_height()

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        ) 

    plt.show() 

def stacked_barplot(data, predictor, target):
    """
    Print the category counts and plot a stacked bar chart

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    count = data[predictor].nunique()
    sorter = data[target].value_counts().index[-1]
    tab1 = pd.crosstab(data[predictor], data[target], margins=True).sort_values(
        by=sorter, ascending=False
    )
    print(tab1)
    print("-" * 120)
    tab = pd.crosstab(data[predictor], data[target], normalize="index").sort_values(
        by=sorter, ascending=False
    )
    tab.plot(kind="bar", stacked=True, figsize=(count + 5, 5))
    plt.legend(
        loc="lower left", frameon=False,
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.show()

def distribution_plot_wrt_target(data, predictor, target):
    """
    Print the distribution plot

    data: dataframe
    predictor: independent variable
    target: target variable
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    target_uniq = data[target].unique()

    axs[0, 0].set_title("Distribution of target for target=" + str(target_uniq[0]))
    sns.histplot(
        data=data[data[target] == target_uniq[0]],
        x=predictor,
        kde=True,
        ax=axs[0, 0],
        color="teal",
        stat="density",
    )

    axs[0, 1].set_title("Distribution of target for target=" + str(target_uniq[2]))
    sns.histplot(
        data=data[data[target] == target_uniq[2]],
        x=predictor,
        kde=True,
        ax=axs[0, 1],
        color="orange",
        stat="density",
    )

    axs[1, 0].set_title("Boxplot w.r.t target")
    sns.boxplot(data=data, x=target, y=predictor, ax=axs[1, 0], palette="gist_rainbow")

    axs[1, 1].set_title("Boxplot (without outliers) w.r.t target")
    sns.boxplot(
        data=data,
        x=target,
        y=predictor,
        ax=axs[1, 1],
        showfliers=False,
        palette="gist_rainbow",
    )

    plt.tight_layout()
    plt.show()

def checking_overfitting_undefitting(y_train, y_train_pred, y_test, y_test_pred):
    """
    Print whether the model is underfit, overfit or good fit.
    
    y_train = training data
    y_train_pred = predictions on training data
    y_test = testing data
    y_test_pred = predictions on testing data
    """
    training_accuracy = accuracy_score(y_train, y_train_pred)
    testing_accuracy = accuracy_score(y_test, y_test_pred)
    if training_accuracy<=0.65:
        print("Model is underfitting.") 
    elif training_accuracy>0.65 and abs(training_accuracy-testing_accuracy)>0.15:
        print("Model is overfitting.")
    else:
        print("Model is not underfitting/overfitting.")

def calculate_classification_metrics(y_true, y_pred, algorithm):
    """
    Return the classification Metrics
    
    y_true = actual values
    y_pred = predicted values
    y_pred_probability = probability values
    algorithm = algorithm name
    """
    accuracy = round(accuracy_score(y_true, y_pred), 3)
    precision = round(precision_score(y_true, y_pred, average='weighted'), 3)
    recall = round(recall_score(y_true, y_pred, average='weighted'), 3)
    f1 = round(f1_score(y_true, y_pred, average='weighted'), 3)
    print("Algorithm: ", algorithm)
    print()
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print()
    cm = confusion_matrix(y_true, y_pred)
    labels = ['Overcast', 'Clear','Foggy']
    plt.figure(figsize=(10, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    return accuracy, precision, recall, f1

# Callback function to avoid overfitting
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('val_accuracy')>0.90) and (logs.get('accuracy')>0.95):
            print("\nValidation and training accuracies are high so cancelling training!")
            self.model.stop_training = True


#  columns analysis
for i in numeric_columns:
    histogram_boxplot(data,i)


for i in categorical_columns:
    if i in ['Daily Summary','Time']:
        pass
    else:
        labeled_barplot(data, i)


# text = ' '.join(data['Daily Summary'].astype(str))
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
# plt.figure(figsize=(10, 6))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()



for i in numeric_columns:
    distribution_plot_wrt_target(data, i, "Summary")


stacked_barplot(data,"Precip Type" , 'Summary')


counts = data["Summary"].value_counts()
total = counts.sum()
percentages = (counts / total) * 100
print(percentages)
print()
print("The classes are satifactory balanced")



input_df = data.drop(columns="Summary", axis=1)
input_df.head()

encoder = LabelEncoder()
y = data["Summary"]
y = encoder.fit_transform(y)

# Checking the mapping of the classes
class_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
for class_label, class_number in class_mapping.items():
    print(f"Class '{class_label}' is labeled as {class_number}")


mapping = {'rain': 0, 'snow': 1}
input_df['Precip Type'] = input_df['Precip Type'].map(mapping)


input_df['Daily Summary Frequency'] = input_df['Daily Summary'].map(input_df['Daily Summary'].value_counts(normalize=True))
input_df.drop(columns=['Daily Summary'], axis=1, inplace=True)
# Checking data
input_df.head()



# Confirming multicollinearity using heatmap
sns.set(style="white")
plt.figure(figsize=(12,8))
sns.heatmap(input_df.corr(), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()


input_df.drop(['Daily Summary Frequency'], axis=1, inplace=True)


input_df.drop(['Temperature (C)'], axis=1, inplace=True)

X = input_df.values
print(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
joblib.dump(scaler, "scaler.pkl")


# Hyperparameter tuning
parameters = {'solver': ['liblinear', 'saga'], 
              'multi_class':['ovr', 'multinomial'],
              'C':[0.001, 0.01, 10.0],
              'penalty': ['l1', 'l2']}
# Model Creation and Training
model_lr = LogisticRegression(n_jobs=-1)
models_lr = GridSearchCV(estimator=model_lr, param_grid=parameters, cv=4)
models_lr.fit(x_train, y_train)
best_parameters = models_lr.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions for train
best_model_lr = models_lr.best_estimator_
y_pred_lr = best_model_lr.predict(x_train)
# Predictions for test
y_pred_lr_new = best_model_lr.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_lr, y_test, y_pred_lr_new)


# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_lr, precision_lr, recall_lr, f1_lr = calculate_classification_metrics(y_test, y_pred_lr_new, "Logistic Regression")


parameters = {'var_smoothing':[1e-9, 1e-8, 1e-10]}
# Model Creation and Training
model_nb = GaussianNB()
models_nb = GridSearchCV(estimator=model_nb, param_grid=parameters, cv=4)
models_nb.fit(x_train, y_train)
best_parameters = models_nb.best_params_
print("Best Hyperparameters:", best_parameters)
print()
# Predictions on training data
best_model_nb = models_nb.best_estimator_
y_pred_nb = best_model_nb.predict(x_train)
# Predictions on test data
y_pred_nb_new = best_model_nb.predict(x_test)
checking_overfitting_undefitting(y_train, y_pred_nb, y_test, y_pred_nb_new)



# Evaluation Metrics Calculation
print("Testing Performance")
accuracy_nb, precision_nb, recall_nb, f1_nb = calculate_classification_metrics(y_test, y_pred_nb_new, "Gaussian NB")


# # Hyperparameter tuning
# parameters = {'loss':['log_loss','perceptron','hinge','squared_epsilon_insensitive'],
#               'penalty': ['l1', 'l2'],
#               'alpha':[0.001,0.01,0.0001],
#               'learning_rate':['optimal','adaptive','invscaling']}
# # Model Creation and Training
# model_svc = SGDClassifier()
# models_svc = GridSearchCV(estimator=model_svc, param_grid=parameters, cv=4)
# models_svc.fit(x_train, y_train)
# best_parameters = models_svc.best_params_
# print("Best Hyperparameters:", best_parameters)
# print()
# # Predictions on train data
# best_model_svc = models_svc.best_estimator_
# y_pred_svc = best_model_svc.predict(x_train)
# # Predictions on test data
# y_pred_svc_new = best_model_svc.predict(x_test)
# checking_overfitting_undefitting(y_train, y_pred_svc, y_test, y_pred_svc_new)


# print("Testing Performance")
# accuracy_svc, precision_svc, recall_svc, f1_svc = calculate_classification_metrics(y_test, y_pred_svc_new, "SVC")


# Hyperparameter tuning
# parameters = {'loss':['log_loss','perceptron','hinge','squared_epsilon_insensitive'],
#               'penalty': ['l1', 'l2'],
#               'alpha':[0.001,0.01,0.0001],
#               'learning_rate':['optimal','adaptive','invscaling']}
# # Model Creation and Training
# model_sgd = SGDClassifier()
# models_sgd = GridSearchCV(estimator=model_sgd, param_grid=parameters, cv=4)
# models_sgd.fit(x_train, y_train)
# best_parameters = models_sgd.best_params_
# print("Best Hyperparameters:", best_parameters)
# print()
# # Predictions on train data
# best_model_sgd = models_sgd.best_estimator_
# y_pred_sgd = best_model_sgd.predict(x_train)
# # Predictions on test data
# y_pred_sgd_new = best_model_sgd.predict(x_test)
# checking_overfitting_undefitting(y_train, y_pred_sgd, y_test, y_pred_sgd_new)


# # Evaluation Metrics Calculation
# print("Testing Performance")
# accuracy_sgd, precision_sgd, recall_sgd, f1_sgd = calculate_classification_metrics(y_test, y_pred_sgd_new, "SGD Classifier")

# # Hyperparameter tuning
# parameters = {'criterion':['gini', 'entropy', 'log_loss'], 
#               'max_depth': [None, 5, 10],
#               'min_samples_split': [None, 2, 5],
#               'splitter':['best','random']}
# # Model Creation and Training
# model_dt = DecisionTreeClassifier()
# models_dt = GridSearchCV(estimator=model_dt, param_grid=parameters, cv=4)
# models_dt.fit(x_train, y_train)
# best_parameters = models_dt.best_params_
# print("Best Hyperparameters:", best_parameters)
# print()
# # Predictions on train data
# best_model_dt = models_dt.best_estimator_
# y_pred_dt = best_model_dt.predict(x_train)
# # Predictions on test data
# y_pred_dt_new = best_model_dt.predict(x_test)
# checking_overfitting_undefitting(y_train, y_pred_dt, y_test, y_pred_dt_new)


# # Evaluation Metrics Calculation
# print("Testing Performance")
# accuracy_dt, precision_dt, recall_dt, f1_dt = calculate_classification_metrics(y_test, y_pred_dt_new, "Decision Tree")





# # Hyperparameter tuning
# parameters = {'max_depth': [None, 5],
#             'class_weight': [None, 'balanced'],
#             'min_samples_split': [None, 2, 5],
#             'criterion':['gini','log_loss','entropy']}
# # Model Creation and Training
# model_et = ExtraTreesClassifier()
# models_et = GridSearchCV(estimator=model_et, param_grid=parameters, cv=4)
# models_et.fit(x_train, y_train)
# best_parameters = models_et.best_params_
# print("Best Hyperparameters:", best_parameters)
# print()
# # Predictions on train data
# best_model_et = models_et.best_estimator_
# y_pred_et = best_model_et.predict(x_train)
# # Predictions on test data
# y_pred_et_new = best_model_et.predict(x_test)
# checking_overfitting_undefitting(y_train, y_pred_et, y_test, y_pred_et_new)



# # Evaluation Metrics Calculation
# print("Testing Performance")
# accuracy_et, precision_et, recall_et, f1_et = calculate_classification_metrics(y_test, y_pred_et_new, "Extra Trees")




# # Results
# print("Testing Performances for Machine Learning Algorithms")
# result = pd.DataFrame({"Algorithms":['Logistic Regression', "Gaussian Naive Bayes", "SVC", "SGD Classifier", "Decision Tree", "KNN","Random Forest", "Extra Trees Classifier", "Bagging Classifier","Gradient Boosting Classifier"],
#                        "Accuracy":[accuracy_lr, accuracy_nb, accuracy_sgd, accuracy_dt, accuracy_et],
#                        "Precision":[precision_lr, precision_nb, precision_sgd, precision_dt, precision_et],
#                        "Recall":[recall_lr, recall_nb, recall_sgd, recall_dt, recall_et],
#                        "F1 Score":[f1_lr, f1_nb, f1_sgd, f1_dt, f1_et]}).set_index('Algorithms')
# print(result)


# Hyperparameters for ANN & RNN
num_classes = 3
epochs = 150
input_dimension = x_train.shape[1]
batch_size = 64
learning_rate = 0.001


# Converting labels to one-hot encoded format
y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes)
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes)


# Reshaping input data for RNN
x_train_reshaped = np.expand_dims(x_train, axis=2)
x_test_reshaped = np.expand_dims(x_test, axis=2)



# Architecture 1: 64-64-128-3 Feed Forward Neural Network
# Defining the ANN architecture
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_dim=input_dimension)) 
model.add(tf.keras.layers.Dense(64, activation='relu')) 
model.add(tf.keras.layers.Dense(128, activation='relu')) 
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dense(num_classes, activation='softmax')) 
# Model Compilation
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# Stopping early to avoid overfitting
stop_callback = myCallback()

# Training
history = model.fit(x_train, y_train_encoded, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test_encoded), callbacks=[stop_callback])

# Plotting training and testing curves
default_size = plt.rcParams['figure.figsize']
fig = plt.figure(figsize=[default_size[0] * 2, default_size[1]])

fig.add_subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label = 'accuracy')          # Train accuracy (blue)
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')  # Valid accuracy (orange)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend(loc='lower right')

fig.add_subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')          # Train loss (blue)
plt.plot(history.history['val_loss'], label='val_loss')  # Valid loss (orange)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend(loc='upper right')
plt.show()



# Evaluating the model on test data
test_loss_0, test_accuracy_0 = model.evaluate(x_test, y_test_encoded, verbose=0)
print('Test Loss:', test_loss_0)
print('Test Accuracy:', test_accuracy_0)
# Saving the model to a file
model.save('FFNN.h5')


