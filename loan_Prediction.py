
import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import mlflow 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics 
from sklearn.model_selection import train_test_split, GridSearchCV
import warnings
warnings.filterwarnings('ignore')



dataset = pd.read_csv('train.xls')
# Select the numerical colums and the categorical columsn seperate 
numeric_columns = dataset.select_dtypes(include=['int64','float64']).columns.tolist()
categorical_columns = dataset.select_dtypes(include=['object']).columns.tolist()
print('Numerical columns :',numeric_columns)
print('Categorical Columns :',categorical_columns)
categorical_columns.remove('Loan_ID')
categorical_columns.remove('Loan_Status')

for col in categorical_columns:
    dataset[col].fillna(dataset[col].mode()[0],inplace=True)

for col in numeric_columns:
    dataset[col].fillna(dataset[col].median(),inplace=True)

# Take care of the outliers 
dataset[numeric_columns]=dataset[numeric_columns].apply(lambda x : x.clip(*x.quantile([0.05,0.95])) )

# LogTransformation annd Domain Processing 
dataset['LoanAmount'] = np.log(dataset['LoanAmount']).copy()
dataset['TotalIncome']= dataset['ApplicantIncome'] + dataset['CoapplicantIncome']
dataset['TotalIncome'] = np.log(dataset['TotalIncome']).copy()

dataset = dataset.drop(columns=['ApplicantIncome','CoapplicantIncome'])

# Label Encoding for the categorical data 
for col in categorical_columns:
    le = LabelEncoder()
    dataset[col] = le.fit_transform(dataset[col])

# encode the target 
dataset['Loan_Status'] = le.fit_transform(dataset['Loan_Status'])

X = dataset.drop(columns=['Loan_Status', 'Loan_ID'])
y = dataset.Loan_Status
RANDOM_SEED = 6

X_train,X_test, y_train,y_test = train_test_split(X,y,random_state=RANDOM_SEED,test_size=0.3)

# Random Forest classifier 
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators':[200,400,700],
    'max_depth': [10,20,30],
    'criterion':['gini','entropy'],
    'max_leaf_nodes':[50,100]
}

grid_forest = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_forest,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)
model_forest = grid_forest.fit(X_train, y_train)

#Logistic Regression

lr = LogisticRegression(random_state=RANDOM_SEED)
param_grid_log = {
    'C': [100, 10, 1.0, 0.1, 0.01],
    'penalty': ['l1','l2'],
    'solver':['liblinear']
}

grid_log = GridSearchCV(
        estimator=lr,
        param_grid=param_grid_log, 
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=0
    )
model_log = grid_log.fit(X_train, y_train)

#Decision Tree

dt = DecisionTreeClassifier(
    random_state=RANDOM_SEED
)

param_grid_tree = {
    "max_depth": [3, 5, 7, 9, 11, 13],
    'criterion' : ["gini", "entropy"],
}

grid_tree = GridSearchCV(
        estimator=dt,
        param_grid=param_grid_tree, 
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=0
    )
model_tree = grid_tree.fit(X_train, y_train)

mlflow.set_experiment('Loan_Prediction')

# model Evaluvate metrics
def model_evaluvate(actual,pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(actual, pred)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f'%auc)
    plt.plot([0,1],[0,1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    # Save plot
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    # Close plot
    plt.close()
    return(accuracy, f1, auc)

def mlflow_logging(model,X,y,name):
    with mlflow.start_run() as run :
        mlflow.set_tracking_uri('http://127.0.0.1:5000/')
        run_id = run.info.run_id 
        
        mlflow.set_tag("run_id",run_id)
        
        pred = model.predict(X)
        
        (accuracy,f1,auc) = model_evaluvate(y,pred=pred)
        
        #Logging the parameters 
        mlflow.log_params(model.best_params_)
        #log the metrics
        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("AUC", auc)
        # Logging artifacts and model
        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.log_artifact("train.xls")
        mlflow.sklearn.log_model(model, name)
        mlflow.end_run()
if __name__ == '__main__':
    mlflow_logging(model_tree, X_test, y_test, "DecisionTreeClassifier")
    mlflow_logging(model_log, X_test, y_test, "LogisticRegression")
    mlflow_logging(model_forest, X_test, y_test, "RandomForestClassifier")