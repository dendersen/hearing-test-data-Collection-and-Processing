from contextlib import contextmanager, redirect_stdout
from io import StringIO
from time import sleep
import streamlit as st
from FunktionsAndMore import*
dtf = pd.read_csv('Data\TempOccupied.csv')
dtf = dtf.rename(columns={"Occupancy":"Y"})
features.append('Temp')

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


output = st.empty()
with st_capture(output.code):
    test_corr(dtf, x='Ratio', y='Y')

output = st.empty()
with st_capture(output.info):
    print("Goodbye")
    print("World")