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
Data = st.text_input('Data to be read: example Data\TempOccupied.csv')
if Data:
  dtf = pd.read_csv(Data)
  st.write(dtf.head())
  fig = dtf_overview(dtf,max_cat=20,figsize=(10,5))
  st.pyplot(fig)

index = st.text_input('Chose Index if needed')
predict = st.text_input('Chose what you want to predict, colum name shoud become "Y"')
if index:
  dtf = dtf.set_index(index)

if predict:
  #We rename the collum we want to predict
  dtf = dtf.rename(columns={predict:"Y"})
  st.write("Distribution of predict")
  st.pyplot(freqdist_plot(dtf, "Y", figsize=(5,3)))
  st.write(dtf.head())
  st.write("Chose data to check wheter or not is Predictive")
  predictive = st.text_input('Collum name of checked')
  if predictive: 
    st.pyplot(bivariate_plot(dtf, x=predictive, y="Y", figsize=(15,5)))
    st.write("The correlation, if the p-value is small enough (<0.05) the null hypothesis of samples means equality can be rejected.")
    output = st.empty()
    with st_capture(output.code):
      test_corr(dtf,x=predictive,y="Y")
    addfeature = st.button('If p-value is <0.05 add to features')
    if addfeature:
      features.append(predictive)
  st.write(features)
  clearFeatures = st.button('Clear features')
  if clearFeatures:
    features.clear()
  FeaturesDone = st.button('All whised features are chosen')
  if FeaturesDone:
    st.write('Here is the chosen features')
    dtf = dtf[features+["Y"]]
    st.write(dtf.head())
    st.write('We can now go to preprocessing')
    output = st.empty()
    with st_capture(output.code):
      data_preprocessing(dtf, y="Y")
    dtf_train, dtf_test = dtf_partitioning(dtf, y="Y", test_size=0.3, shuffle=False)
    #We show what we just did abowe^
    output = st.empty()
    with st_capture(output.code):
      dtf_partitioning(dtf, y="Y", test_size=0.3, shuffle=False)
    #Show training data
    st.write('Training data is:')
    st.write(dtf_train.head(3))
    #Show test data
    st.write('Test data is:')
    st.write(dtf_test.head(3))