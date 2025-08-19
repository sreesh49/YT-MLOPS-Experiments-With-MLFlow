
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns   # type: ignore

import dagshub  # type: ignore
dagshub.init(repo_owner='sreesh49', repo_name='YT-MLOPS-Experiments-With-MLFlow', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/sreesh49/YT-MLOPS-Experiments-With-MLFlow.mlflow')
#mlflow.set_tracking_uri("http://127.0.0.1:5000") #local host server 


#Load wine dataset

wine = load_wine()

x=wine.data
y=wine.target

#train test split

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=42)

#Define the params for RF model

max_depth = 5
n_estimators=10

#mention the experiment below 

mlflow.set_experiment('YT-MLOPS-Exp1')

with mlflow.start_run():

    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)

    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

   #creating a confusion matrix plot

    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')

    #save plot
    plt.savefig("Confusion-matrix.png")

    #log artifacts using mlflow

    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    print(accuracy)