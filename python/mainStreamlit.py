import streamlit as st
st.header('A hearing test project:')

# st.sidebar.success("Select what to do")

st.markdown(
"""
Hello👋
\n
This project is made to give insight into data communication and introduces a way to gather data about an individual's hearing. 
The app that you are currently looking at is made to be able to gather data, visualize the data and includes a way to prepare some 
allready preprossesed data for a simple neural network made with sklearn. 
\n
It's important to note that the projekt in it self is a prototype and that the neural network isn't a totally finished product. 👈
"""
)

st.header('The different pages')

st.write(f'<h1 style="color:#FFFFFF;font-size:20px;">{"DataCollect🎧"}</h1>', unsafe_allow_html=True)

st.markdown(
"""
Welcome to the data collection :smile:
\n
This is where in the app you take the test and gather the data from a given test result. 
It is made to include an interface for the person you are testing called "Input data here". 
Here the person is asked to give name, gender, age, hearing loss (if a person has had a hearing 
loss confirmed), headphone time and place of the test. 
\n
Next is where you, the tester, are supposed to input what two frequency tones you want the test 
to be between and how many tones you want (typically 22 is a good number). When all mentioned abowe 
is inputted, the test can be started by a simple click on the button "start test". 
\n
!DO NOT DO ANYTHING UNTIL A TEXT SAYING "TEST IS DONE" SHOWS UP!
"""
)

st.write(f'<h1 style="color:#FFFFFF;font-size:20px;">{"DataShow📈"}</h1>', unsafe_allow_html=True)

st.markdown(
"""
Welcome to the data visualisation :smile:
\n
You just took some tests, and now you want to show what the results show. To see the stats of all data choose, 
in the dropdown box "how to show data", "overall stats" to be shown. 
\n
If you instead chose "individual stats", you are shown a plot that we will descripe the interactions you can have with.
\n
1. "Chose ID"
- Choose the ID (specifik person) you want to show results off.
\n
2. "Chose ear"
- Choose which ear you want to see the results off.
\n
For a description of the all the plots read the code explanation
\n
"""
)

st.write(f'<h1 style="color:#FFFFFF;font-size:20px;">{"DataSortAndFix🛠️"}</h1>', unsafe_allow_html=True)

st.markdown(
"""
Welcome to the data manipulation part and where to choose what to predict with the neural network 🧮
\n
Now that you have visualized the data, you might want to work with it to predict something like hearing loss or age. 
\n 
1. Chose data
First thing first, to choose what data you want to read input the link in the box called "Data to be read". 
The head of the data will then be shown together with a plot visualising the distribution of categoric, numeric and NaN data.
\n
2. Chosse id and data to predict
Now choose what id  you want the data to follow (you dont have to choose an id), and then what you want to predict. 

This will give you a plot of the distrubution of data you want to predict.
\n
3. Find if something is predictive
You will then be asked to check if something is predictive. To do this input name of collum e.g "Frekvens". 
Next you will be shown a plot and a correlation evaluation is shown. Based on that do what is told above.
The next 
\n
4. Last step
Click the last button "Save this data" when you have checked the data for faults.
\n
"""
)

st.write(f'<h1 style="color:#FFFFFF;font-size:20px;">{"NeuralNetwork🤖"}</h1>', unsafe_allow_html=True)

st.markdown(
"""
Welcome to the neural network, where you can try to see if our model can predict something :smile:
\n
This will not be explained so much, basically, you can choose if you want a binary network or not
\n  
"""
)