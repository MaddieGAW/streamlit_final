import streamlit as st
import pandas as pd
import joblib

# Load data
df5 = pd.read_csv('df5.csv')
df5.dropna(inplace=True)

# Define the main function to create and run the app
def main():
    st.title('Project Success Prediction')

    # Add input fields for each feature
    project_size = st.number_input('Project Size (USD)', value=df5['project_size_USD_calculated'].mean())
    startyear = st.number_input('Start Year', value=df5['startyear'].median())
    evalyear = st.number_input('Evaluation Year', value=df5['evalyear'].median())
    eval_lag = st.number_input('Evaluation Lag', value=df5['eval_lag'].median())
    project_duration = st.number_input('Project Duration', value=df5['project_duration'].median())

    donor = st.selectbox('Donor', df5['donor'].unique())
    country_code = st.selectbox('Country Code', df5['country_code_WB'].unique())
    region = st.selectbox('Region', df5['region'].unique())
    colonial_relations = st.selectbox('Colonial Relations', df5['colonial_relations'].unique())
    sector_code = st.selectbox('Sector Code', df5['sector_code'].unique())
    office_presence = st.selectbox('Office Presence', df5['office_presence'].unique())
    external_evaluator = st.selectbox('External Evaluator', df5['external_evaluator'].unique())

    # Load the trained model
    model_path = 'model.joblib'
    model = joblib.load(model_path)

    # Prepare new data for prediction
    new_data = pd.DataFrame({
        'project_size_USD_calculated': [project_size],
        'startyear': [startyear],
        'evalyear': [evalyear],
        'eval_lag': [eval_lag],
        'project_duration': [project_duration],
        'donor': [donor],
        'country_code_WB': [country_code],
        'region': [region],
        'colonial_relations': [colonial_relations],
        'sector_code': [sector_code],
        'office_presence': [office_presence],
        'external_evaluator': [external_evaluator]
    })

    # Predict the success of the project
    if st.button('Predict'):
        prediction = model.predict(new_data)
        st.subheader('Prediction')
        if prediction[0] == 1:
            st.write('The project is predicted to be successful.')
        else:
            st.write('The project is predicted to not be successful.')

# Run the app
if __name__ == "__main__":
    main()
