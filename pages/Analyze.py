import streamlit as st
import pandas as pd 

import altair as alt 
st.markdown("# Contrails Analysis Tool  🎈")
st.sidebar.markdown("# Analyze 🎈")

#DATE_COLUMN = 'date/time'
DATA_URL = './aay_testset1687372286-summary.csv'
# ('https://s3-us-west-2.amazonaws.com/'        'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
         
@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
  #  lowercase = lambda x: str(x).lower()
  #  data.rename(lowercase, axis='columns', inplace=True)
  #  data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data = load_data(10000)
# Notify the reader that
dep_airport_list = data['dep_airport_name'].unique()
arr_airport_list = data['arr_airport_name'].unique()
dep_airport_option = ''
arr_airport_option = '' 

def get_arr_airport_list():
    if dep_airport_option is not None:
        fd = data[data['dep_airport_name'] == dep_airport_option] 
        return fd['arr_airport_name'].unique()
    else:
        return arr_airport_list
    
def get_dep_airport_list():
    if arr_airport_option != '':
        fd = data[data['arr_airport_name'] == arr_airport_option] 
        return fd['dep_airport_name'].unique()
    else:
        return dep_airport_list
    
    
dep_airport_option = st.selectbox(
    'Departure Airport',
     get_dep_airport_list())
     
arr_airport_option = st.selectbox(
     'Arrival Airport',
     get_arr_airport_list())

selected_route = dep_airport_option + " --> " + str(arr_airport_option)
'You selected: ', selected_route 
# dep_airport_option + " --> " arr_airport_option 

filtered_data = data[data['dep_airport_name'] == dep_airport_option] 
filtered_data = filtered_data[filtered_data['arr_airport_name'] == arr_airport_option]
num_flights = len(filtered_data)
st.subheader(f'Avaialable Flights {num_flights}:00')
#st.map(filtered_data)

st.write(filtered_data)

c = alt.Chart(data).mark_circle().encode(x='initial_contrail_length_km', y='persistent_contrail_length_km', size='total_contrail_EF', color='fms_route_name', tooltip=['aircraft_model', 'fms_route_name', 'dep_airport_name', 'arr_airport_name', 'total_contrail_EF', 'fuel_burn']) #, 'b', 'c'])
 
st.altair_chart(c, use_container_width=True)
