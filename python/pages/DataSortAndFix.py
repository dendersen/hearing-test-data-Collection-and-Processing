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

st.title('Check Data')
Data = st.text_input('Data to be read: example Data\editResultStorage.csv')
if Data:
  dtf = pd.read_csv(Data)
  st.write(dtf.head())
  fig = dtf_overview(dtf,max_cat=20,figsize=(10,5))
  st.pyplot(fig)
  output=st.empty()
  with st_capture(output.code):
    print('Categerocial = blue, Numerical/DateTime = red, NaN = white')

index = st.text_input('Chose Index if needed')
predict = st.text_input('Chose what you want to predict')
if index:
  dtf = dtf.set_index(index)

if predict:
  st.write("Distribution of predict")
  st.pyplot(freqdist_plot(dtf, predict, figsize=(10,5)))
  st.write(dtf.head())
  st.write("Chose data to check wheter or not is Predictive")
  predictive = st.text_input('Collum name of checked')
  if predictive:
    dtfNoIndex = dtf.reset_index()
    st.pyplot(bivariate_plot(dtf, x=predictive, y=predict, figsize=(15,5)))
    
    if (utils_recognize_type(dtf, predictive, 20) == "num") & (utils_recognize_type(dtf, predict, 20) == "num"):
      ## joint plot
      st.pyplot(sns.jointplot(x=predictive, y=predict, data=dtfNoIndex, dropna=True, kind='reg', height=int((15+5)/2)))
    
    st.write("The correlation, if the p-value is small enough (<0.05) the null hypothesis of samples means equality can be rejected.")
    output = st.empty()
    with st_capture(output.code):
      test_corr(dtf,x=predictive,y=predict)
    addfeature = st.button('If p-value is <0.05 add to features')
    if addfeature:
      features.append(predictive)
  st.write(features)
  clearFeatures = st.button('Clear features')
  if clearFeatures:
    features.clear()
  st.title('If all whised features are chosen, continue')
  st.write('Here is the chosen features')
  dtf = dtf[features+[predict]]
  st.write(dtf.head())
  st.write('We can now go to preprocessing, value we want to predict is now Y')
  dtf = dtf.rename(columns={predict:"Y"})
  output = st.empty()
  with st_capture(output.code):
    data_preprocessing(dtf, y="Y")
  dtf_train, dtf_test = dtf_partitioning(dtf, y="Y", test_size=0.1, shuffle=False)
  #We show what we just did abowe^
  output = st.empty()
  with st_capture(output.code):
    dtf_partitioning(dtf, y="Y", test_size=0.1, shuffle=False)
  #Show training data
  st.write('Training data is:')
  st.write(dtf_train.head(3))
  #Show test data
  st.write('Test data is:')
  st.write(dtf_test.head(3))
  #Check if rebalance is needed
  st.write('Check if rebalance is needed')
  output = st.empty()
  with st_capture(output.code):
    check = rebalance(dtf_train, y="Y", balance=None)
  #We show the final result, when data is fixed
  dtf_train = pop_columns(dtf_train, ["Y"], where="end")
  dtf_test = dtf_test[dtf_train.columns]
  st.write('Result of fixed data, that we scale')
  st.write(dtf_train.head())
  st.write(dtf_test.head())
  st.write('Result of scaled data')
  scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
  dtf_train, scaler = scaling(dtf_train, y="Y", scalerX=scaler, task="classification")
  dtf_test, _ = scaling(dtf_test, y="Y", scalerX=scaler, fitted=True)
  st.write(dtf_train.head())
  st.write(dtf_test.head())
  st.pyplot(dtf_overview(dtf_train))
  
  #Save data, so that it can be used in a neural network
  Save = st.button('Save this data')
  if Save:
    with open('Data\ScaleDataTrain.csv','w') as f:
      dtf_train.to_csv(f,header=True)
    with open('Data\ScaleDataTest.csv','w') as f:
      dtf_test.to_csv(f,header=True)

  # #We chose the data we wish to train with, X_train should not contain the output values so we remove them
  # X_names = dtf_train.drop(predict, axis=1).columns.tolist()
  # X_train = dtf_train.drop(predict, axis=1).values
  # y_train = dtf_train[predict].values
  # X_test = dtf_test.drop(predict, axis=1).values
  # y_test = dtf_test[predict].values
  # st.write(X_test.shape)
  # st.title('Neural Network')
  # #Defines the metrics
  # def Recall(y_true, y_pred):
  #   true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  #   possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
  #   recall = true_positives / (possible_positives + K.epsilon())
  #   return recall
  # #Finds the precision of the prediction
  # def Precision(y_true, y_pred):
  #   true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
  #   predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
  #   precision = true_positives / (predicted_positives + K.epsilon())
  #   return precision
  # #We chose to make a F1-score, a metric that combines Precision and Recall, because we allso want to monitor this
  # def F1(y_true, y_pred):
  #   precision = Precision(y_true, y_pred)
  #   recall = Recall(y_true, y_pred)
  #   return 2*((precision*recall)/(precision+recall+K.epsilon()))
  
  # # Deep neural network with fully connected layers
  # #Here you assume an input dataset of N features and 1 binary target variable
  # NFeatures = int(st.number_input('Add exstra features',value=5))
  # #If extra units are needed
  # AddExstraUnits = int(st.number_input('Add exstra units',value=6))
  # AU = AddExstraUnits
  # #we then make the model
  # model = models.Sequential(name="DeepNN", layers=[
  #   #hidden layer 1
  #   layers.Dense(
  #     name="h1",
  #     input_dim=NFeatures,                  #N features as the input
  #     units=int(round((NFeatures+1)/2)+AU), #The amount of units we want
  #     activation='relu',                    #max(0,x)=x1_x>0
  #   ),
  #   layers.Dropout(name='drop1', rate=0.2),
    
  #   #hidden layer 2
  #   layers.Dense(
  #     name="h2",
  #     units=int(round((NFeatures+1)/4)+AU),
  #     activation='relu',
  #   ),
  #   layers.Dropout(name="drop2", rate=0.2),
    
  #   #hidden layer 3
  #   layers.Dense(
  #     name="h3",
  #     units=int(round((NFeatures+1)/4)+AU),
  #     activation='relu',
  #   ),
  #   layers.Dropout(name="drop3", rate=0.2),
    
  #   #hidden layer 4
  #   layers.Dense(
  #     name="h4",
  #     units=int(round((NFeatures+1)/4)+AU),
  #     activation='relu',
  #   ),
  #   layers.Dropout(name="drop4", rate=0.2),
    
  #   #layer output
  #   layers.Dense(
  #     name="output",
  #     units=1,
  #     activation='sigmoid' #f(x) = 1 / (1 + e^(-x))
  #   ),
  # ])
  
  # #We allso need to compile the neural network, as we now do
  # model.compile(optimizer='adam',           #I chose, as in the totorial, to use the Adam optimizer, a replacement optimization algorithm for gradient descent
  #               loss='binary_crossentropy', #This classification problem compares each of the predicted probabilities to the actual class output.
  #               metrics=['accuracy',F1]     #Input the metrics here
  # )
  
  # #We print the summary
  # output = st.empty()
  # with st_capture(output.code):
  #   model.summary()
  # #We show the network viasual
  # st.pyplot(visualize_nn(model, description=True, figsize=(10,8)))
  
  # #Run model
  # Run = st.button('Run model on data')
  # if Run:
  #   st.write('Model results')
  #   output = st.empty()
  #   model, predicted_prob, predicted = fit_dl_classif(X_train, y_train, X_test, model, batch_size=32, epochs=500, threshold=0.5)
  #   output = st.empty()
  #   with st_capture(output.code):
  #     evaluate_classif_model(y_test, predicted, predicted_prob, figsize=(25,5))
  #   st.pyplot(evaluate_classif_model(y_test, predicted, predicted_prob, figsize=(25,5)))
  #   #Explainer
  #   st.write('Explainer')
  #   i = 2
  #   output = st.empty()
  #   with st_capture(output.code):
  #     print("True:", y_test[i], "--> Pred:", int(predicted[i]), "| Prob:", np.round(np.max(predicted_prob[i]), 2))
  #   st.pyplot(explainer_shap(model, X_names, X_instance=X_test[i], X_train=X_train, task="classification", top=10))