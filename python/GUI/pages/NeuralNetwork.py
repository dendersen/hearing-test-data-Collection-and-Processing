import streamlit as st
from FunktionsAndMore import*
from matplotlib.colors import ListedColormap
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
option = st.selectbox('Chose if network i binary or not', ('Binary', 'Not binary'))

if option == "Binary":
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
  NFeatures = int(st.number_input('Add exstra features, features indicated abowe',value=len(dtf_train.columns)-1))
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
  #We show the network visual
  st.pyplot(visualize_nn(model, description=True, figsize=(10,8)))
  #Run model
  Run = st.button('Run model on data')
  if Run:
    st.write('Model results')
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
    st.write('Model values 2D')
    st.pyplot(plt.show())
    if NFeatures == 2:
      st.pyplot(plot2d_classif_model(X_train, y_train, X_test, y_test, model, annotate=False, figsize=(10,5)))
    #We show the facit
    st.write('Real values 2D')
    RealValues = dtf_train.values
    for i in range(len(RealValues)):
      if RealValues[i,NFeatures] == 1:
        plt.scatter(RealValues[i,0],RealValues[i,1],c="green")
      else:
        plt.scatter(RealValues[i,0],RealValues[i,1],c="black")
    st.pyplot(plt.show())

if option == "Not binary":
  st.write('Shown down below is the different labels')
  st.write(np.unique(y_test))
  #We store labels
  label_names = np.unique(y_test)
  #We make the model:
  # Set random seed
  tf.random.set_seed(19)
  # Create the model
  #Here you assume an input dataset of N features and 1 binary target variable
  NFeatures = int(st.number_input('Add exstra features',value=len(dtf_train.columns)-1))
  labels = st.number_input("Write the amount of labels",value=len(np.unique(y_test)))
  #If extra units are needed
  AddExstraUnits = int(st.number_input('Add exstra units',value=6))
  AU = AddExstraUnits
  # Output is 1 of the x possible labels
  model = tf.keras.Sequential([  
    tf.keras.layers.Dense(4+AU, activation="relu"),
    tf.keras.layers.Dense(4+AU, activation="relu"),
    tf.keras.layers.Dense(4+AU, activation="relu"),
    tf.keras.layers.Dense(4+AU, activation="relu"),
    tf.keras.layers.Dense(4+AU, activation="relu"),
    tf.keras.layers.Dense(4+AU, activation="relu"),
    tf.keras.layers.Dense(4+AU, activation="relu"),
    tf.keras.layers.Dense(labels, activation="softmax")
  ])
  
  # Use SparseCategoricalCrossentropy if data isn't 
  # normalized or one-hot encoded
  model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=["accuracy"])
  
  #Run model
  Run = st.button('Run model on data')
  if Run:
    # Fit the model
    history = model.fit(X_train, 
                          y_train,
                          epochs=500,
                          validation_data=(X_test, y_test))
    #We print the summary
    output = st.empty()
    with st_capture(output.code):
      model.summary()
    
    #visualize the model
    st.pyplot(visualize_nn(model, description=True, figsize=(10,8)))
    
    # Model loss curve
    pd.DataFrame(history.history).plot()
    plt.title("model Loss Curve")
    st.pyplot(plt.show())
    
    #Evaluate with test data
    st.write("Evaluation with test data")
    output = st.empty()
    loss, acc = model.evaluate(X_test, y_test)
    with st_capture(output.code):
      print(f"Model Loss (Test set) : {loss}")
      print(f"Model accuracy (Test set) : {acc}")
    
    st.write('Output of network shown here, not alway 2D')
    # Prediction Probabilities
    y_prob = model.predict(X_test) 
    ColorList = ["black","green","red","orange","blue","pink","olive","cyan","purple","gold","crimson"]
    for i in range(len(X_test)):
      plt.scatter(x=X_test[i][0],y=X_test[i][1],c=ColorList[label_names[tf.argmax(y_prob[i])]])
    st.pyplot(plt.show())
    # show model parameters
    if NFeatures == 2:
      colors = {np.unique(y_test)[i]:ColorList[i] for i in range(len(np.unique(y_test)))}
      print(colors)
      X1, X2 = np.meshgrid(np.arange(start=X_test[:,0].min()-1, stop=X_test[:,0].max()+1, step=0.01),
                           np.arange(start=X_test[:,1].min()-1, stop=X_test[:,1].max()+1, step=0.01))
      label_names = np.unique(y_test)
      Y = np.array([])
      b = model.predict(np.array([X1.ravel(), X2.ravel()]).T)
      for i in range(len(np.array([X1.ravel(), X2.ravel()]).T)):
        Y = np.append(Y,[label_names[tf.argmax(b[i])]])
      Y = Y.reshape(X1.shape)
      fig, ax = plt.subplots(figsize=(10,8))
      ax.contourf(X1, X2, Y, alpha=0.5, levels=np.arange(Y.max() + 2) - 0.5, cmap=ListedColormap(list(colors.values())))
      ax.set(xlim=[X1.min(),X1.max()], ylim=[X2.min(),X2.max()], title="Classification regions")
      for i in np.unique(y_test):
        ax.scatter(X_test[y_test==i, 0], X_test[y_test==i, 1], c=colors[i], label="true "+str(i))  
      plt.legend()
      st.pyplot(plt.show())
    
    st.write('Real values')
    RealValues = dtf_train.values
    for i in range(len(RealValues)):
        plt.scatter(RealValues[i,0],RealValues[i,1],c=ColorList[int(RealValues[i,NFeatures])])
    st.pyplot(plt.show())