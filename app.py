import streamlit as st
import json
from openai import AzureOpenAI
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import threading

df = pd.read_csv('canteen_menu.csv')
if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True
# def display_recipe_recommendations(df):
    
#     st.sidebar.subheader('Step 2: Analyze the recipes based on your likes')
#     diet_filter = st.sidebar.radio("Select Your Diet type", ('paleo', 'vegan', 'keto', 'No specific diet'))
#     cuisine_filter = st.sidebar.multiselect('Choose Your Preferred Cuisines', df.Cuisine_type.unique(), ('Chinese', 'American', 'Korean'))
#     done = st.sidebar.button("Done")
    
#     # Filter by eating habits
#     if diet_filter != 'No specific diet':
#         df = df[df.Diet_type == diet_filter]

#     # Filter by cuisine
#     df = df[df.Cuisine_type.isin(cuisine_filter)]
#     st.subheader('The table below displays the recommended recipes based on your diets and preferred cuisines.')
#     st.write(df[['Restaurant', 'Recipe_name', 'Cuisine_type', 'Carbs(kcal)']])

# Initial message content as a JSON object
initial_content = {
    "isNextState": False,
    "resp": "Hi! I am Aliang, your persoanl fitness instructor and nutritionist. Since You might intend to keep fit but don't know where to start, I'd like to help you design your individual fitness plan and dietary plan based on your personal fitness characteristic and eating habits. It's nice to meet you and let's get started! 😊",
    "data": ""
}

initial_content['prompt'] = json.dumps(initial_content)
st.sidebar.subheader('Step 1: Enter your basic information')
gender = st.sidebar.radio("Select You Gender",('male', 'female', 'prefer not to say'),index=None)
age = st.sidebar.text_input("Your age:")
height = st.sidebar.text_input("Your height (in cm):")
weight = st.sidebar.text_input("Your weight (in kg):")
target_weight = st.sidebar.text_input("Your target weight (in kg):")

states = {
    'Greeting': {
        'next': 'CollectExerciseInfo',
        'description': "Greet the user and introduce the chatbot's purpose.",
        'collectedDataName': None  # No data collected in this state
    },
   'CollectExerciseInfo': {
        'next': 'CollectNumOfMealsDaily',
        'description': "Ask the user for exercise standards and ability, activity level and relative information.",
        'collectedDataName': 'exerciseInfo'  # Collecting exercise information
    },
    'CollectNumOfMealsDaily': {
        'next': 'CollectDataOnAllMeals',
        'description': "Ask the user for the number of meals he or she consumed yesterday.",
        'collectedDataName': 'numOfMealsDaily'  # Collecting Num of Meals consumed yesterday
    },
    'CollectDataOnAllMeals': {
        'next': 'ProvideAdvice',
        'description': 'Collect as much data as possible about all the meals consumed by the user yesterday such as what they had for each meal,including information about the meal type, ingredients, and portion sizes,and that they provide details on all the meals they said they have consumed. Remind them to provide any dietary restrictions or allergies',
        'collectedDataName': 'CollectedDataOnAllMeals'  # Collecting data on all meals that day
    },
    'ProvideAdvice': {
        'next': 'ProvideCanteenAdvice',
        'description': "Based on 'CollectedDataOnAllMeals', calculate an estimate on the total amount of calories consumed by the user that day. Identify the breakdown of the food based on fats, carbs and protein. Provide advice on whether the meal consumed was healthy, and provide suggestions on how the user can improve his daily diet to mantain his health",
        'collectedDataName': None  # No data collected in this state
    },
    'ProvideCanteenAdvice': {
        'next': 'ProvidePersonalPlan',
        'description': "Provide the canteen advice to user.",
        'collectedDataName': None  # No data collected in this state
    },
    'ProvidePersonalPlan':{
        'next': 'Unhandled',
        'description': "Based on all the information collected, provide a overall fitness plan to the user for three days, including the composition of each meal, the exercise plan for each day, and briefly introduce how might this plan helps the user",
        'collectedDataName': None  # No data collected in this state
    },
    'Unhandled': {
        'next': None,
        'description': "Handle any unrelated or unclear inputs by guiding the user back to the conversation or asking for clarification.",
        'collectedDataName': None  # Varies based on the user input
    }
}


def next_state(current_state):
    """
    Determines the next state based on the current state.

    Parameters:
    - current_state: The current state of the conversation.

    Returns:
    - The name of the next state.
    """
    # Get the next state from the current state's information
    next_state = states[current_state]['next']

    # If there's no next state defined, it means we're at the end of the flow or in an unhandled situation
    if not next_state:
        return None

    return next_state


def create_model_prompt(user_content):
    current_state = st.session_state['current_state']
    state_description = states[current_state]['description']
    next_state = states[current_state]['next']
    next_state_description = states[next_state]['description'] if next_state else states[current_state]['description']

    # User basic info
    data= {
        "gender":gender,
        "age":age,
        "height":height,
        "weight":weight,
        "target weight":target_weight 
    }
    basic_info = json.dumps(data)
    
    collected_data_json = json.dumps(st.session_state.get('user_data', {}))
    prompt = f"""
    Answer with a JSON object in a string without line breaks, with the following fields:
    - isNextState: Boolean value indicating whether the goal of the current state is satisfied.
    - resp: Textual response for the user.
    - data: String value representing the current collected data, if applicable.
    
    You are a fitness instructor and nutritionist chatbot designed to help people improve their nutrition and keep fit.
    The user basic information is: {basic_info}. The current state of your conversation with the user is {current_state}, which means {state_description}.
    If the goal of the current state is satisfied, the next state is {next_state}, which means {next_state_description}.
    The new response from the user is: {user_content}.
    The collected data is: {collected_data_json}.
    
    Pay attention to some invalid input and decide whether the goal of the current state is satisfied. If yes, set isNextState to true; otherwise, set it to false. 
    If isNextState is true and the current state is about collecting data, put the collected data value (only the value of the current data collection goal) in the data field; otherwise, leave the data field empty.
    Provide your response to the user in the resp field. 
    If isNextState is true, proceed with the action of the next state (such as asking questions or give the weight loss guidance); otherwise, try to reach the goal by giving a response.
    """
    

    return prompt

def get_response_from_model(client):
    # Send the prompt to the model
    # if st.session_state['current_state'] == "ProvideCanteenAdvice":
        # isCanteenAdviceState = True
    if not st.session_state.clicked and st.session_state['current_state'] == "ProvideCanteenAdvice":
        # display_recipe_recommendations(df)
        response = {
            "isNextState":False,
            "resp":"Did you like our suggestions? Press Done if you do.",
            "data":""
        }
        response = json.dumps(response)
        response_data = json.loads(response)

        return response_data
    if age=="" or gender==None or height=="" or weight=="" or target_weight=="":
        response = {
            "isNextState":False,
            "resp":"Sorry, could you please enter your basic information in left sidebar first before we start?",
            "data":""
        }
        response = json.dumps(response)
        response_data = json.loads(response)

        return response_data
    else:
        msgs = [{"role": m['role'], "content": m['content']['prompt']} for m in st.session_state.messages]
        response = client.chat.completions.create(
            model=model_name,
            messages=msgs,
        )

        # Parse the model's response
        model_response = response.choices[0].message.content

        # to see if the response is a JSON string
        print(model_response)

        # Assuming the model's response is a JSON string; parse it
        response_data = json.loads(model_response)

        return response_data
    
    
if 'current_state' not in st.session_state:
    st.session_state['current_state'] = 'Greeting'
    st.session_state['user_data'] = {}

with st.sidebar:
    openai_api_key = st.text_input("Azure OpenAI API Key", key="chatbot_api_key", type="password")
    "[Get an Azure OpenAI API key](https://itsc.hkust.edu.hk/services/it-infrastructure/azure-openai-api-service)"

model_name = "gpt-35-turbo"

st.title("💬 Aliang: a fitness instructor and nutritionist chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": initial_content}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"]['resp'])

if user_resp := st.chat_input():
    if not openai_api_key:
        st.info("Please add your Azure OpenAI API key to continue.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": {'prompt': create_model_prompt(user_resp), 'resp': user_resp}}
    )
    st.chat_message("user").write(user_resp)


    # setting up the OpenAI model
    client = AzureOpenAI(
        api_key=openai_api_key,
        api_version="2023-12-01-preview",
        azure_endpoint="https://hkust.azure-api.net/",
    )
    
    model_resp = get_response_from_model(client)

    # state transition
    if model_resp['isNextState']:
        if states[st.session_state['current_state']]['collectedDataName']:
            st.session_state['user_data'][states[st.session_state['current_state']]['collectedDataName']] = model_resp[
                'data']
        st.session_state['current_state'] = next_state(st.session_state['current_state'])

    # ensure the consistency
    model_resp['prompt'] = json.dumps(model_resp)

    st.session_state.messages.append({"role": "assistant", "content": model_resp})
    st.chat_message("assistant").write(model_resp['resp'])

if st.session_state["current_state"]=="ProvideCanteenAdvice":
    st.sidebar.subheader('Step 2: Analyze the recipes based on your likes')
    diet_filter = st.sidebar.radio("Select Your Diet type", ('paleo', 'vegan', 'keto', 'No specific diet'))
    cuisine_filter = st.sidebar.multiselect('Choose Your Preferred Cuisines', df.Cuisine_type.unique(), ('Chinese', 'American', 'Korean'))
    st.sidebar.button("Done", on_click=click_button)
    # Filter by eating habits
    if diet_filter != 'No specific diet':
        df = df[df.Diet_type == diet_filter]

    # Filter by cuisine
    df = df[df.Cuisine_type.isin(cuisine_filter)]
    st.subheader('The table below displays the recommended recipes based on your diets and preferred cuisines.')
    st.write(df[['Restaurant', 'Recipe_name', 'Cuisine_type', 'Carbs(kcal)']])