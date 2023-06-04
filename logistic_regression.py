import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import mean_squared_error


# preprocess data
#read in data
trainpath="train.csv"
testpath="test.csv"
trainX=pd.read_csv(trainpath,usecols=[*range(1,14),23])
#trainX.info()
trainX=trainX.to_numpy()
trainy=pd.read_csv(trainpath,usecols=[0])
trainy=trainy.to_numpy()
testX=pd.read_csv(testpath,usecols=[*range(0,13),22])
#testX.info()
testX=testX.to_numpy()
N=len(trainX)
testN=len(testX)

imp = IterativeImputer(max_iter=100000, random_state=0)
imp.fit(trainX)
trainX=imp.transform(trainX)
imp = IterativeImputer(max_iter=100000, random_state=0)
imp.fit(testX)
testX=imp.transform(testX)
print("impute missing values done")


# Train a separate linear regression model for each class
models = []
for c in np.unique(trainy):
    # Create a binary target variable (1 if the instance belongs to the class, 0 otherwise)
    binary_y = np.where(trainy == c, 1, 0)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(trainX, binary_y)
    models.append(model)

# Predict the class label probabilities for new instances
i = 0
pred = np.zeros(testN)*(-1)
for instance in testX:
    predicted_probabilities = []
    for model in models:
        # Predict the probability of belonging to the class
        probability = model.predict([instance])
        predicted_probabilities.append(probability)
    
    # Apply softmax function to convert regression outputs into probabilities
    probabilities = np.exp(predicted_probabilities) / np.sum(np.exp(predicted_probabilities))
    
    # Choose the class with the highest probability
    predicted_class = np.argmax(probabilities)
    pred[i] = predicted_class
    i+=1
    print(f"Instance {i} is predicted to belong to class {predicted_class}")

print(pred)

ids=list(range(N,N+testN))
with open("lgrpred.csv","w") as f:
    f.write('id,Danceability\n')

with open("lgrpred.csv", "a") as f:
    for each_id, row in zip(ids,pred):
        line = "%d" %each_id + ","+"%f" %row + "\n"
        f.write(line)

# Evaluate the performance on the training dataset using mean squared error (MSE)
y_pred = []
for instance in trainX:
    predicted_probabilities = []
    for model in models:
        probability = model.predict([instance])
        predicted_probabilities.append(probability)
    
    probabilities = np.exp(predicted_probabilities) / np.sum(np.exp(predicted_probabilities))
    predicted_class = np.argmax(probabilities)
    y_pred.append(predicted_class)

mse = mean_squared_error(trainy, y_pred)
print(f"Mean Squared Error (MSE) on training dataset: {mse}")