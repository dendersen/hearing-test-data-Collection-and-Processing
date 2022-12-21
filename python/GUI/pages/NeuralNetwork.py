import streamlit as st
from FunktionsAndMore import*
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)
@contextmanager
def st_capture(output_func):
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield

#We read the data that we made
dtf_test = pd.read_csv('Data\ScaleDataTest.csv', index_col=0)
dtf_train = pd.read_csv('Data\ScaleDataTrain.csv', index_col=0)

#We chose the data we wish to train with, X_train should not contain the output values so we remove them
X_names = dtf_train.drop("Y", axis=1).columns.tolist()
X_train = dtf_train.drop("Y", axis=1).values
y_train = dtf_train["Y"].values
X_test = dtf_test.drop("Y", axis=1).values
y_test = dtf_test["Y"].values

st.title('Neural Network')
st.write(X_test.shape)
#Defines the metrics
def Recall(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  recall = true_positives / (possible_positives + K.epsilon())
  return recall
#Finds the precision of the prediction
def Precision(y_true, y_pred):
  true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  precision = true_positives / (predicted_positives + K.epsilon())
  return precision
#We chose to make a F1-score, a metric that combines Precision and Recall, because we allso want to monitor this
def F1(y_true, y_pred):
  precision = Precision(y_true, y_pred)
  recall = Recall(y_true, y_pred)
  return 2*((precision*recall)/(precision+recall+K.epsilon()))

# Deep neural network with fully connected layers
#Here you assume an input dataset of N features and 1 binary target variable
NFeatures = int(st.number_input('Add exstra features, features indicated abowe',value=5))
#If extra units are needed
AddExstraUnits = int(st.number_input('Add exstra units',value=6))
AU = AddExstraUnits
#we then make the model
model = models.Sequential(name="DeepNN", layers=[
  #hidden layer 1
  layers.Dense(
    name="h1",
    input_dim=NFeatures,                  #N features as the input
    units=int(round((NFeatures+1)/2)+AU), #The amount of units we want
    activation='relu',                    #max(0,x)=x1_x>0
  ),
  layers.Dropout(name='drop1', rate=0.2),
  
  #hidden layer 2
  layers.Dense(
    name="h2",
    units=int(round((NFeatures+1)/4)+AU),
    activation='relu',
  ),
  layers.Dropout(name="drop2", rate=0.2),
  
  #hidden layer 3
  layers.Dense(
    name="h3",
    units=int(round((NFeatures+1)/4)+AU),
    activation='relu',
  ),
  layers.Dropout(name="drop3", rate=0.2),
  
  #hidden layer 4
  layers.Dense(
    name="h4",
    units=int(round((NFeatures+1)/4)+AU),
    activation='relu',
  ),
  layers.Dropout(name="drop4", rate=0.2),
  
  #layer output
  layers.Dense(
    name="output",
    units=1,
    activation='sigmoid' #f(x) = 1 / (1 + e^(-x))
  ),
])

#We allso need to compile the neural network, as we now do
model.compile(optimizer='adam',           #I chose, as in the totorial, to use the Adam optimizer, a replacement optimization algorithm for gradient descent
              loss='binary_crossentropy', #This classification problem compares each of the predicted probabilities to the actual class output.
              metrics=['accuracy',F1]     #Input the metrics here
)

#We print the summary
output = st.empty()
with st_capture(output.code):
  model.summary()
#We show the network viasual
st.pyplot(visualize_nn(model, description=True, figsize=(10,8)))

#Run model
Run = st.button('Run model on data')
if Run:
  st.write('Model results')
  output = st.empty()
  model, predicted_prob, predicted = fit_dl_classif(X_train, y_train, X_test, model, batch_size=32, epochs=500, threshold=0.5)
  output = st.empty()
  with st_capture(output.code):
    evaluate_classif_model(y_test, predicted, predicted_prob, figsize=(25,5))
  st.pyplot(evaluate_classif_model(y_test, predicted, predicted_prob, figsize=(25,5)))
  #Explainer
  st.write('Explainer')
  i = 2
  output = st.empty()
  with st_capture(output.code):
    print("True:", y_test[i], "--> Pred:", int(predicted[i]), "| Prob:", np.round(np.max(predicted_prob[i]), 2))
  st.pyplot(explainer_shap(model, X_names, X_instance=X_test[i], X_train=X_train, task="classification", top=10))
  #We show the models predictions
  Y = model.predict(X_train)
  for i in range(len(Y)):
    if Y[i] > 0.5:
      plt.scatter(X_train[i,0],X_train[i,1],c="green")
    else:
      plt.scatter(X_train[i,0],X_train[i,1],c="black")
  st.write('Model values')
  st.pyplot(plt.show())
  #We show the facit
  st.write('Real values')
  st.pyplot(Scatter_plot(dtf_train,'Temp','Humidity','Y'))