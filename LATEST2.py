import base64
from email import header
from email.base64mime import header_length
from operator import index
from pickle import NONE
from tkinter import font
from tkinter.tix import COLUMN
from turtle import width
from typing import Sized
import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score
from sklearn. ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os
import pipreqs as pr


st.set_page_config(
    page_title="Data Science Assignment 1",
    page_icon="bar_chart",
    layout="wide",
)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('wallpaper.jpg')

with st.sidebar:
    selected1 = option_menu(
        menu_title = "Main Menu",
        options = ["Home Page", "Main Dashboard", "Random Forest",  "Naive Bayes",  "Decision Trees", "Libraries Used"],
        icons = ["house", "book"],) 


if selected1 == "Libraries Used":
    
    with st.spinner('Generating list...'):
        st.title('You have selected the "Libraries Used" page ')
        st.subheader('This page contains a list of all libraries/modules used in this web app')
    
        st.markdown("'<h1 style='text-align: center; font-size:30px; line-height:40%'>Modules imported:",unsafe_allow_html=True)

        #Generates requirements.text file
        CurDir=os.getcwd()
        cmd='pipreqs --force'
        os.system(cmd)


        txt=open('requirements.txt','r')
        ReqRead=txt.read().splitlines()
        for line in ReqRead:
            line_text="<h1 style='text-align: center; color: #37c860; font-size:20px; line-height:30%'>{}".format(line)
            st.write(line_text,unsafe_allow_html=True)


if selected1 == "Home Page":

    st.title(f"You have selected {selected1}")
    st.title("<h1 style='text-align: center; color: #4092BB; font-size:75px;'>The Dataset Bike_Buyers</h1>", "The Dataset (Bike_Buyers)")
    st.markdown("##")
    st.markdown("##")
    st.markdown("##")

    st.subheader('The dataset that we choose is Bike_Buyer.csv on this dataset it consists of 1000 rows of data with 13 unique columns, we downloaded this dataset from a website called Kaggle.com where it has a variety of dataset to choose from. Using the dataset ive chosen, producing a best dashboard application where it consists of attractive visualiztion.')
    st.write("")
    st.subheader("Below are the complete dataset that we have chosen for this assignment, it consists of 1001 rows of data with 13 unique columns. There are 2 data types involves in this dataset (Numerical)(Character)")
    bike = pd.read_csv("bike_buyers_dataset.csv")
    bike

if selected1 == "Main Dashboard":
    st.title(f"You have selected {selected1}")
    selected11 = option_menu(
    menu_title = "My Dashboard",
    options = ["1st visual", "2nd visual", "3rd visual", "4th visual", "5th visual", "6th visual"],
    icons = ["house", "book", "envelope"],
    orientation = "horizontal")
    if selected11 == "1st visual":
        st.title(f"You have selected {selected11}")
        st.subheader(" Below visualization is a bargraph of the selected column within the dataset agains the column ID")
        #1st Visualization
        df1 = pd.read_csv("bike_buyers_dataset.csv")

        selected_x_var1 = st.selectbox('What do you want the x variable to be?',
        ['Marital_Status', 'Gender', 'Education', 'Occupation', 'Home_Owner', 'Commute Distance', 'Region', 'Cars'])
        selected_y_var1 = st.selectbox('What do you want the y variable to be?',
        ['Income', 'Children', 'Age',])

        sns.set_style('darkgrid')
        fig, ax = plt.subplots()
        ax = sns.factorplot(data = df1, x = selected_x_var1,
        y = selected_y_var1, hue = 'Occupation')
        plt.xlabel(selected_x_var1)
        plt.ylabel(selected_y_var1)
        plt.title("Factor plot based on the variables chosen against Occupation")
        st.pyplot(ax.figure)

    if selected11 == "2nd visual":
        st.title(f"You have selected {selected11}")
    #2nd Visualization
        df2 = pd.read_csv("bike_buyers_dataset.csv")

        selected_x_var2 = st.selectbox('What do you want the x-axis to be?',
        ['Marital_Status', 'Gender', 'Education', 'Occupation', 'Home_Owner', 'Commute Distance', 'Region', 'Cars'])
        selected_y_var2 = st.selectbox('What do you want the y-axis  to be?',
        ['Income', 'Children', 'Age',])

        sns.set_style('darkgrid')
        fig, ax = plt.subplots()
        ax = sns.barplot(data = df2, x = selected_x_var2,
        y = selected_y_var2, hue = 'Cars')
        plt.xlabel(selected_x_var2)
        plt.ylabel(selected_y_var2)
        plt.title("Barplot of the chosen variables against Cars")
        st.pyplot(ax.figure)

    if selected11 == "3rd visual":
        st.title(f"You have selected {selected11}")
    #3rd Visualization
        df3 = pd.read_csv("bike_buyers_dataset.csv")

        selected_x_var3 = st.selectbox('What do you want the xaxis to be?',
        ['Marital_Status', 'Gender', 'Education', 'Occupation', 'Home_Owner', 'Commute Distance', 'Region', 'Cars'])
        selected_y_var3 = st.selectbox('What do you want the yaxis  to be?',
        ['Income', 'Children', 'Age',])

        sns.set_style('darkgrid')
        fig, ax = plt.subplots()
        ax = sns.lineplot(data = df3, x = selected_x_var3,
        y = selected_y_var3)
        plt.xlabel(selected_x_var3)
        plt.ylabel(selected_y_var3)
        plt.title("Lineplot of the chosen variables")
        st.pyplot(ax.figure)
    
    if selected11 == "4th visual":
        st.title(f"You have selected {selected11}")
    #4th Visualization
        selected_var4 = st.selectbox('What do you want the x.. to be?',
        ['Marital_Status', 'Gender', 'Education', 'Occupation', 'Home_Owner', 'Commute Distance', 'Region', 'Cars'])
        df4 = pd.read_csv("bike_buyers_dataset.csv")
        fig4 = px.histogram(df4, x = selected_var4, color = (selected_var4), marginal = "rug", hover_data = df4.columns, width = 1000, height = 700)
        st.plotly_chart(fig4)

    if selected11 == "5th visual":
        st.title(f"You have selected {selected11}")
    #5th Visualization
        selected_var5 = st.selectbox('What do you want the .x to be?',
        ['Income', 'Children', 'Age',])
        labels = (selected_var5)
        df5 = pd.read_csv("bike_buyers_dataset.csv")
        df5[selected_var5] = df5[selected_var5].astype(float)
        df5["Region"] = df5["Region"].astype("str")
        fig5 = px.area(df5, x = selected_var5, y = "Region", color = "Region", width = 1000, height = 700)
        st.plotly_chart(fig5)

    if selected11 == "6th visual":
        st.title(f"You have selected {selected11}")
        st.subheader(f"This is a pie chart that shows the % within the column chosen")
        selected_var6 = st.selectbox('What do you want the .x to be?',
        ['Marital_Status', 'Gender', 'Education', 'Occupation', 'Home_Owner', 'Commute Distance', 'Region', 'Cars'])
        labels = ['selected_x_var5']
    #6th Visualization
        df6 = pd.read_csv("bike_buyers_dataset.csv")
        fig6 = px.pie(df6, selected_var6, labels = labels, title='Differentiate the Age/Children/Income based on the Region', width = 1000, height = 700)
        st.plotly_chart(fig6)


#1 (Random Forest)
if selected1 == "Random Forest":
    col1, mid, col2 = st.columns([1,2,20])
    with col1:
        st.image('rfc.png', width=100)
    with col2:
        st.title(f"_Random Forest Classifier_")

    st.subheader("This app uses 3 inputs to predict the customer's Gender.")
  
    from email.headerregistry import UniqueSingleAddressHeader
    from turtle import distance
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from sklearn.metrics import accuracy_score, classification_report, precision_score

    from sklearn.ensemble import RandomForestClassifier

    from sklearn.model_selection import train_test_split

    import pickle

    import streamlit as st

    from tabulate import tabulate

    df = pd.read_csv('bike_buyers_dataset.csv')
    df.dropna(inplace=True)
    output = df['Gender']

    features = df[['Marital_Status','Age', 'Education']]
    features = pd.get_dummies(features)
    output, uniques = pd.factorize(output)
    print('features')
    print(features)
    print('output')
    print(output)
    print('uniques')
    print(uniques)

    x_train, x_test, y_train, y_test = train_test_split(
        features, output, test_size=.8)
    rfc = RandomForestClassifier(random_state=15)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    score = accuracy_score(y_pred, y_test)
    precision = precision_score(y_test, y_pred)

    print("Classification Report:\n ", classification_report (y_test, y_pred))
    print('Our accuracy score for this model is {}'.format(score))

    #import pickle
    rf_pickle = open('random_forest_bike_buyers.pickle','wb')
    pickle.dump(rfc, rf_pickle)
    rf_pickle.close()
    output_pickle = open('output_bike_buyers.pickle','wb')
    pickle.dump(uniques, output_pickle)
    output_pickle.close()

    rf_pickle = open('random_forest_bike_buyers.pickle', 'rb')
    map_pickle = open('output_bike_buyers.pickle', 'rb')
    rfc = pickle.load(rf_pickle)
    unique_bike_buyers_mappping = pickle.load(map_pickle)

    Marital_Status = st.selectbox('Marital Status', options=['Single','Married'])
    Age = st.number_input('Age',min_value=df['Age'].min(),max_value=df['Age'].max())
    #Education = st.selectbox('Education', options=['Bachelors','Partial College','High School','Partial High School','Graduate Degree'])
    Occupation = st.selectbox('Occupation', options=['Clerical','Management','Manual','Professional','Skilled Manual'])

    st.success('The user inputs are: {}'.format(
    [Marital_Status, Age, Occupation])) #Page 16

    rf_pickle = open('random_forest_bike_buyers.pickle','rb')
    map_pickle = open('output_bike_buyers.pickle','rb')
    rfc = pickle.load(rf_pickle)
    unique_bike_buyers_mappping = pickle.load(map_pickle)

    rf_pickle.close()
    map_pickle.close()

    ms_single, ms_married = 0, 0
    if Marital_Status == 'Single':
        ms_single = 1
    elif Marital_Status == 'Married':
        ms_married = 1

    # education_bachelors, education_partialcollege, education_highschool, education_partialhighschool, education_graduatedegree = 0, 0, 0, 0, 0
    # if Education == 'Bachelors':
    #     education_bachelors = 1
    # elif Education == 'Partial College':
    #     education_partialcollege = 1
    # elif Education == 'High College':
    #     education_highschool = 1
    # elif Education == 'Partial High School':
    #     education_partialhighschool = 1 
    # elif Education == 'Graduate Degree':
    #     education_graduatedegree = 1

    occupation_clerical, occupation_management, occupation_manual, occupation_professional, occupation_skilledmanual = 0, 0, 0, 0, 0
    if Occupation == 'Clerical':
        occupation_clerical = 1
    elif Occupation == 'Management':
        occupation_management = 1
    elif Occupation == 'Manual':
        occupation_manual = 1
    elif Occupation == 'Professional':
        occupation_professional = 1
    elif Occupation == 'Skilled Manual':
        occupation_skilledmanual = 1

    new_prediction = rfc.predict([[ms_single,ms_married,Age,occupation_clerical,
    occupation_management,occupation_manual,occupation_professional,
    occupation_skilledmanual]])

    prediction_gender = unique_bike_buyers_mappping[new_prediction][0]
    st.title("_Result of Model Evaluation_")
    st.subheader('We predict your Gender is {}'.format(prediction_gender))
    
    st.title('_Precision Score_ and _Accuracy Score_')
    data = [['{0:0.3f}'.format(precision),'{0:0.3f}'.format(score)]]
    col_names = ["         Precision Score         ", "           Accuracy Score           "]
    st.text(tabulate(data, headers=col_names, tablefmt="fancy_grid", showindex="always", numalign="center"))

    fig, ax = plt.subplots()
    ax = sns.barplot(x=rfc.feature_importances_, y=features.columns, palette='Paired', edgecolor='black', hatch='///')
    st.title('_Predicting your Gender_')
    plt.title('Which features are the most important for Gender prediction?', size=9)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    fig.savefig('feature_important.png')

    st.subheader("We used a machine learning (Random Forest) model to " "predict the gender, the features used in this predictionaire ranked by relative importance below")
    image = Image.open("feature_important.png")
    new_image = image.resize((1000,750))
    st.image(new_image)



#2 (Naive Bayes)
if selected1 == "Naive Bayes":
    col1, mid, col2 = st.columns([1,3,20])
    with col1:
        st.image('naivebayes.png', width=130)
    with col2:
        st.title(f"_Naive Bayes Classifier_")

    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    import pickle
    import streamlit as st
    from tabulate import tabulate

    df = pd.read_csv('bike_buyers_dataset.csv')
    df.dropna(inplace=True)
    output = df['Purchased Bike']

    ###Data Transformation###
    features= df[[ 'Income', 'Children','Region', 'Age']]
    features=pd.get_dummies(features)
    output, uniques = pd.factorize(output)
    print('features')
    print(features)
    print('Output')
    print(output)
    print('uniques')
    print(uniques)

    ###Training and Testing###
    x_train, x_test, y_train, y_test = train_test_split(
    features, output, test_size=.8,random_state=15)
    nb = MultinomialNB()
    nb.fit(x_train, y_train) 
    y_pred = nb.predict(x_test)
    recall = recall_score(y_test, y_pred)
    score = accuracy_score(y_pred, y_test) 
    print("Classification Report:\n ", classification_report (y_test, y_pred))
    print('Our accuracy score for this model is {}'.format(score))

    ###Model and Outputs Dumping###
    nb_pickle = open('Naive_Bayes_Bike_Buyers.pickle', 'wb')
    pickle.dump(nb, nb_pickle) 
    nb_pickle.close() 
    PBoutput_pickle = open('PBoutput_BikeBuyers.pickle', 'wb')
    pickle.dump(uniques, PBoutput_pickle)
    PBoutput_pickle.close()

    ##Streamlit##
    st.subheader("This app uses 4 inputs to predict whether the described customer would purchase a bike or not.")

    nb_pickle = open('Naive_Bayes_Bike_Buyers.pickle', 'rb') 
    map_pickle = open('PBoutput_BikeBuyers.pickle', 'rb')
    nb = pickle.load(nb_pickle) 
    unique_BikePurchases_mapping = pickle.load(map_pickle) 

    Income=st.number_input('Income',min_value=df['Income'].min(),max_value=df['Income'].max())
    Age=st.number_input('Age',min_value=df['Age'].min(),max_value=df['Age'].max())
    Children=st.select_slider('No. of Children',options=[0,1,2,3,4,5])
    Region=st.selectbox('Region',options=['Europe', 'Pacific', 'North America'])

    st.success('The user inputs are: {}'.format(
    [Income, Age, Children, Region]))

    Region_Europe, Region_North_America, Region_Pacific = 0,0,0
    if Region=='Europe':
        Region_Europe=1
    elif Region=='Pacific':
        Region_Pacific=1
    elif Region=='North America':
        Region_North_America=1

    new_prediction = nb.predict([[Income,Children,Age,Region_Europe,Region_North_America,Region_Pacific]])

    prediction_bike_purchase = unique_BikePurchases_mapping[new_prediction][0] 
    st.subheader('Prediction Outcome = {}'.format(new_prediction))
    st.subheader('Mapped prediction = {}'.format(prediction_bike_purchase))

    if prediction_bike_purchase=='Yes':
        outcome='will'
    elif prediction_bike_purchase=='No':
        outcome='will not'

    st.title('_Result of Model Evaluation_')
    st.subheader('We predict a bike {} be purchased by the described customer'.format(outcome))

    st.title('_Recall Score and Accuracy Score_')
    data = [['{0:0.3f}'.format(recall),'{0:0.3f}'.format(score)]]
    col_names = ["         Recall Score         ", "           Accuracy Score           "]
    st.text(tabulate(data, headers=col_names, tablefmt="fancy_grid", showindex="always", numalign="center"))



#3 (Decision Trees)
if selected1 == "Decision Trees":
    col1, mid, col2 = st.columns([1,2,20])
    with col1:
        st.image('decisiontrees.png', width=100)
    with col2:
        st.title(f"_Decision Trees Classifier_")

    import pandas as pd
    from sklearn.metrics import accuracy_score, precision_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import export_text
    import pickle
    import streamlit as st
    from tabulate import tabulate


    df = pd.read_csv('bike_buyers_dataset.csv')
    df.dropna(inplace=True)
    output = df['Occupation']

    ###Data Transformation###
    features= df[[ 'Education','Region', 'Age']]
    features=pd.get_dummies(features)
    output, uniques = pd.factorize(output)
    print('features')
    print(features)
    print('Output')
    print(output)
    print('uniques')
    print(uniques)

    x_train, x_test, y_train, y_test = train_test_split(
    features, output, random_state=0)
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train) 
    y_pred = dt.predict(x_test)
    precision = precision_score(y_test, y_pred, average='macro')
    score = accuracy_score(y_pred, y_test) 

    print("Classification Report:\n ", classification_report (y_test, y_pred))
    print('Our accuracy score for this model is {}'.format(score))

    dt_pickle = open('Decision_Tree_Bike_Buyers.pickle', 'wb')
    pickle.dump(dt, dt_pickle)
    dt_pickle.close() 
    DToutput_pickle = open('DToutput_BikeBuyers.pickle', 'wb')
    pickle.dump(uniques, DToutput_pickle)
    DToutput_pickle.close()

    st.subheader("This app uses 3 inputs to predict the customer's Education.")

    dt_pickle = open('Decision_Tree_Bike_Buyers.pickle', 'rb') 
    map_pickle = open('DToutput_BikeBuyers.pickle', 'rb')
    dt = pickle.load(dt_pickle) 
    unique_Education_mapping = pickle.load(map_pickle) 

    Age=st.number_input('Age',min_value=df['Age'].min(),max_value=df['Age'].max())
    Education=st.selectbox('Education',options=['Bachelors','Partial College','High School','Partial High School','Graduate Degree'])
    Region=st.selectbox('Region',options=['Europe', 'Pacific', 'North America'])

    st.success('The user inputs are: {}'.format(
    [Age, Education, Region]))

    Region_Europe, Region_North_America, Region_Pacific = 0,0,0
    if Region=='Europe':
        Region_Europe=1
    elif Region=='Pacific':
        Region_Pacific=1
    elif Region=='North America':
        Region_North_America=1

    education_bachelors, education_partialcollege, education_highschool, education_partialhighschool, education_graduatedegree = 0, 0, 0, 0, 0
    if Education == 'Bachelors':
        education_bachelors = 1
    elif Education == 'Partial College':
        education_partialcollege = 1
    elif Education == 'High College':
        education_highschool = 1
    elif Education == 'Partial High School':
        education_partialhighschool = 1 
    elif Education == 'Graduate Degree':
        education_graduatedegree = 1

    new_prediction = dt.predict([[Age, education_bachelors,education_graduatedegree,education_highschool,
    education_partialcollege,education_partialhighschool, Region_Europe, Region_Pacific, Region_North_America]])

    prediction_Education = unique_Education_mapping[new_prediction][0]
    st.title("_Result of Model Evaluation_")
    st.subheader('We predict your Education is: {}'.format(prediction_Education))

    st.title("_Precision Score and Accuracy Score_")
    data = [['{0:0.3f}'.format(precision),'{0:0.3f}'.format(score)]]
    col_names = ["         Precision Score         ", "           Accuracy Score           "]
    st.text(tabulate(data, headers=col_names, tablefmt="fancy_grid", showindex="always", numalign="center"))

    fig, ax = plt.subplots()
    ax = sns.barplot(y=dt.feature_importances_, x=features.columns, palette='Set2', edgecolor='black', hatch='')
    st.title('_Predicting your Education_')
    plt.title('Which features are the most important for Education prediction?')
    plt.ylabel('Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation = 70)
    plt.tight_layout()
    fig.savefig('feature_important.png')

    st.subheader("We used a machine learning (Decision Trees) model to " "predict the Eduction, the features used in this predictionaire ranked by relative importance below")
    image = Image.open("feature_important.png")
    new_image = image.resize((1000,750))
    st.image(new_image)






























