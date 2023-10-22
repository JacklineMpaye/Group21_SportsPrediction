import streamlit as st
import pandas as pd
import pickle
from urllib.request import urlopen

loaded_model = pickle.load(urlopen('https://github.com/Priscile2/Group21_SportsPrediction/releases/download/v0.1/best_model.pkl'))
scaler = pickle.load(open('scaler.pkl', 'rb') )

# Create a Streamlit app
st.set_page_config(
    page_title="FIFA Player Rating Prediction",
    page_icon="⚽️",
    layout="wide",
    initial_sidebar_state="expanded",  # Open the sidebar by default
    
)
st.title("Welcome to FIFA Player Rating Prediction")
st.write("Enter player attributes to predict their overall rating.")


# Define input fields for user input
movement_reactions = st.number_input("Movement Reactions", min_value=0, max_value=100, value=50)
mentality_composure = st.number_input("Mentality Composure", min_value=0, max_value=100, value=50)
passing = st.number_input("Passing", min_value=0, max_value=100, value=50)
potential = st.number_input("Potential", min_value=0, max_value=100, value=50)
release_clause_eur = st.number_input("Release Clause (in EUR)", min_value=0, value=0)
dribbling = st.number_input("Dribbling", min_value=0, max_value=100, value=50)
wage_eur = st.number_input("Wage (in EUR)", min_value=0, value=0)
power_shot_power = st.number_input("Power Shot Power", min_value=0, max_value=100, value=50)
value_eur = st.number_input("Value (in EUR)", min_value=0, value=0)
mentality_vision = st.number_input("Mentality Vision", min_value=0, max_value=100, value=50)
attacking_short_passing = st.number_input("Attacking Short Passing", min_value=0, max_value=100, value=50)
button = st.button('Player Rating')
reset = st.button('Reset')

if button:
    input_data = {
    "movement_reactions": movement_reactions,
    "mentality_composure": mentality_composure,
    "passing": passing,
    "potential": potential,
    "release_clause_eur": release_clause_eur,
    "dribbling": dribbling,
    "wage_eur": wage_eur,
    "power_shot_power": power_shot_power,
    "value_eur": value_eur,
    "mentality_vision": mentality_vision,
    "attacking_short_passing": attacking_short_passing
    }

    #print(input_data)
    input_df = pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df)
    #print(scaled_input)

    prediction = loaded_model.predict(scaled_input)
    st.write(f"The player rating is {int(prediction[0])}")
    st.write(f"The confidence score of the model is 92%")