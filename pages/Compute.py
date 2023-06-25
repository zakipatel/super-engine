
import streamlit as st
import xml.etree.ElementTree as et 
import pandas as pd
#from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from io import StringIO

from pycontrails import Flight, Aircraft
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.humidity_scaling import ConstantHumidityScaling

st.markdown("### Flight Plan Contrail Measurement Tool") 
st.sidebar.markdown("# Compute ❄️")

#defaults
engine_efficiency_input = 0.45
non_volatile_emissions_index_input =  1500000000000000 
n_engine_input = 2
thrust_setting_input = 0.22  
wingspan_dict = {
  "A320": 37.57,
  "B737": 36.0,
  "A350": 65,
  "A380": 80
}

pressure_levels = [300, 250, 200]
params = {
    "process_emissions": False,
    "verbose_outputs": True,
  #  "dt_integration": np.timedelta64(10, "m"),
    "humidity_scaling": ConstantHumidityScaling(rhi_adj=0.98), 
    }

filename_input = "AAY-1142-121100z-KPIT-R03-arinc633"   # has contrails
uploaded_file = st.file_uploader("Upload a ARINC 633 XML file")

import folium 

def map_route(waypoints): 
    m = folium.Map(width=240, height=140)

    polyline = []
    lat = waypoints['latitude']
    lon = waypoints['longitude']
    wp = waypoints['waypoint_name']
    #r = waypoints['rhi']
    #ef = waypoints['ef']
    alt = waypoints['altitude']
    i = 0
    for w in wp:
        point = (lat[i], lon[i])
        name = wp[i]
        #efv = efval[i]
        #if r[i] is not None: 
        #    rhi = r[i] 
            #altv = alt[i]
            #else:
            #    rhi = 0
            #      label = name + " .... RHI: " + str(rhi) + str(efval)
        #if efv > 0:
        folium.Marker(point) #popup= label,
        #icon=folium.Icon(color="red", icon='exclamation', prefix='fa')).add_to(m)
        #elif rhi > 1:                                               #icon_color='white', icon='male', angle=0, prefix='fa'
        #folium.Marker(point, popup= label, opacity=0.5, icon=folium.Icon(color="blue", icon='cloud', prefix='fa')).add_to(m)
        
        polyline.append(point)
        i = i+1
        if i == len(waypoints):
            break
    # add the lines
    folium.PolyLine(polyline, weight=5, opacity=1).add_to(m)

    # create optimal zoom
    df = pd.DataFrame(polyline).rename(columns={0:'Lat', 1:'Lon'})[['Lat', 'Lon']]
    #mls[0]).rename(columns={0:'Lon', 1:'Lat'})[['Lat', 'Lon']]
    sw = waypoints[['latitude', 'longitude']].min().values.tolist()
    ne = waypoints[['latitude', 'longitude']].max().values.tolist()
    m.fit_bounds([sw, ne])

    return m

def parse(filename_input):
    print("****************************************************************************************************")
    print(len(filename_input))
    print("****************************************************************************************************")

    #print(filename_input[1])
    xtree = et.ElementTree(et.fromstring(filename_input)) # .parse(filename_input)
    print("****************************************************************************************************")
    #print(type(xtree))
   # .parse(filename_input) # "./" + filename_input )
    xroot = xtree.getroot()
   

    #M633 Supplementary Header
    header_s_633 = xroot[1]
    flight_origin_date = header_s_633[0].get('flightOriginDate')
    flight_number = header_s_633[0][0][1][0].text 
    dep_airport_name = header_s_633[0][1].get('airportName')
    arr_airport_name = header_s_633[0][2].get('airportName')

    aircraft_reg = header_s_633[1].get('aircraftRegistration')
    aircraft_subtype = header_s_633[1][0].get('airlineSpecificSubType')
    aircraft_model = header_s_633[1][0][0].text  

    #M633 FlightInfo
    flight_info = xroot[2]
    aircraft_reg = header_s_633[1].get('aircraftRegistration')
    call_sign = flight_info.get('aTCCallsign')

    #M633 FlightPlan Header
    flight_plan_header = xroot[4]

    performance_factor = flight_plan_header[1].text # ('PerformanceFactor')
    fuel_flow_avg = flight_plan_header[2][0][0].text
    fuel_flow_holding = flight_plan_header[2][1][0].text
    fms_route_name = flight_plan_header[3].get('fMSRouteName') #.text
    route_name = flight_plan_header[3].get('routeName')
    wind_direction_avg = flight_plan_header[3][0][0][0].text
    wind_speed_avg = flight_plan_header[3][0][1][0].text
    ground_distance = flight_plan_header[3][9][0].text
    air_distance = flight_plan_header[3][10][0].text
    greatcircle_distance = flight_plan_header[3][11][0].text

    #M633 Weight Header
    weight_header = xroot[6]
    takeoff_weight = int(weight_header[3][0][0].text)#/2.205
    landing_weight = int(weight_header[4][0][0].text)#/2.205
    burn = int(takeoff_weight) - int(landing_weight)

    # Waypoints
    waypoints = xroot[7]
    df_cols = ["sequence_no", "waypoint_name", 
               "latitude", "longitude", 
               "altitude", "elapsed_time", 
              # "mach_number", 
               "true_airspeed", 
               #"indicated_airspeed",
               "engine_efficiency", "time", "burn_off", "c"]
    rows = []

    for wp in waypoints:
    #for wp in xroot.iter('Waypoint'): 
        sequence_no = int(wp.get('sequenceId'))
        waypoint_name = wp.get('waypointName')
        lat = float(wp[0].get('latitude'))/3600
        long = float(wp[0].get('longitude'))/3600
        altitude, burn_off, fuel_flow, mach_number, true_airspeed, indicated_airspeed = 0.0,0.0, 0.0, 0.0, 0.0, 0.0
        seconds = 0
        timestamp = '2023-05-23T04:36:00' # datetime.today()
        date_format = '%Y-%m-%dT%H:%M:%S'
        altitude = None

        for child in wp:
            if child.tag == '{http://aeec.aviation-ia.net/633}Altitude':
                altitude = (float(child[0][0].text)*100)/3.218  if child[0][0] is not None else 0
            if child.tag == '{http://aeec.aviation-ia.net/633}BurnOff':
                burn_off = float(child[0][0].text) #/2.205
            if child.tag == '{http://aeec.aviation-ia.net/633}TimeOverWaypoint':
                timestamp = child[0][0].text
                #print(timestamp)
            if child.tag == '{http://aeec.aviation-ia.net/633}TimeFromPreviousWaypoint':
                split = child[0][0].text.split("H")
                minutes = split[1] 
                seconds = float(minutes.split("M")[0])*60
                #else:
                #    fuel_flow = 0
                #time = time - timedelta(days=13)
            if child.tag == '{http://aeec.aviation-ia.net/633}MachNumber':
                mach_number = float(child[0][0].text) # if mach_no_node is not None else 0
            if child.tag == '{http://aeec.aviation-ia.net/633}TrueAirSpeed':
                true_airspeed = float(child[0][0].text)*0.514 # if mach_no_node is not None else 0
            if child.tag == '{http://aeec.aviation-ia.net/633}IndicatedAirSpeed':
                indicated_airspeed = float(child[0][0].text)*0.514 # if mach_no_node is not None else 0

        rows.append({"sequence_no": sequence_no,
                 "waypoint_name": waypoint_name,
                 "latitude": lat,
                 "longitude": long, 
                 "altitude": altitude,
                 "elapsed_time": seconds, 
                 "time": timestamp, #datetime.strptime(timestamp, date_format) , 
                 "burn_off": burn_off,
                 #"mach_number": mach_number,
                 "true_airspeed": true_airspeed, 
                #"indicated_airspeed": indicated_airspeed,
                 "engine_efficiency": 0.0
                })

    # Set Takeoff Weight 
    rows[0]['c'] = takeoff_weight
    # Set Initial Altitude
    rows[0]['altitude'] =0
    # Set Final Altitude
    rows[len(rows)-1]['altitude']=0

    out_df = pd.DataFrame(rows, columns = df_cols)
    #out_df.info()
    new_col = 'c'

    def apply_func_decorator(func):
        prev_row = {}
        def wrapper(curr_row, **kwargs):
            val = func(curr_row, prev_row)
           # print(val)
           # print(prev_row)
            prev_row.update(curr_row)
           # print(prev_row)
            prev_row[new_col] = val
           # print(prev_row)
           # print(val)
            return val
        return wrapper

    @apply_func_decorator
    def aircraft_mass(curr_row, prev_row):
        #print(curr_row)
        #print(prev_row)
        return prev_row.get("c", takeoff_weight) - curr_row["burn_off"]

    #+ curr_row['b'] + prev_row.get('c', 0)
    out_df["aircraft_mass"] = out_df.apply(aircraft_mass, axis=1)

    # fuel flow 
    #"Fuel mass flow rate (kg s-1)": "fuel_flow",
    #Kilogram Per Second (kg/s) is a unit in the category of Mass flow rate. 
    #It is also known as kilogram/second, kilograms per second. 
    #This unit is commonly used in the SI unit system. Kilogram Per Second (kg/s) has a dimension of MT-1 where M is mass, and T is time.

    def fuel_flow(df):
        if df["burn_off"] > 0 and df["elapsed_time"] > 0:
            return df["burn_off"] / df["elapsed_time"]
        else:
            return 0

    out_df["fuel_flow"] = out_df.apply(fuel_flow, axis=1)

    # Engine Efficiency

    F=0.22  # Thrust 
    Q=43.24  # 

    def engine_efficiency(df):
        V = int(df["indicated_airspeed"])*0.514
        fuel_flow = df["fuel_flow"]
        if V > 0 and fuel_flow > 0:
            efficiency = (F*V)/(Q*fuel_flow)
            return efficiency
        else:
            return 0

    out_df["engine_efficiency"] =  engine_efficiency_input # out_df.apply(engine_efficiency, axis=1)

    #out_df.info()
    out_df = out_df.drop('c', axis=1)
    cleaned_df = out_df.drop_duplicates(subset='time', keep="first")
    #cleaned_df = cleaned_df[cleaned_df.altitude != 0]

    #cleaned_df.describe()
    # max_altitude = ?
    # max_speed = ? 

    # demo synthetic flight

    flight_attrs = {
        "flight_id": call_sign,
        # set constants along flight path
        #"true_airspeed": true_airspeed_input, 
        "thrust": thrust_setting_input, 
        "nvpm_ei_n": non_volatile_emissions_index_input,
    }
    df = cleaned_df.interpolate(method='linear', limit_direction='forward', axis=0)
    df = df.fillna(0)

    #print(flight_attrs)
    # wingspan_dictionary =
    wingspan_input = 37.57  
    if aircraft_model in wingspan_dict:
        wingspan_input =  wingspan_dict[aircraft_model]

    aircraft = Aircraft(aircraft_type=aircraft_model, wingspan=wingspan_input, n_engine=n_engine_input)
    flight = Flight(df, aircraft=aircraft, attrs=flight_attrs)
    time = (
        pd.to_datetime(flight["time"][0]).floor("H"),
        pd.to_datetime(flight["time"][-1]).ceil("H"), #+ pd.Timedelta("10H"),
    )
    
    return flight, time, aircraft 

from streamlit_folium import st_folium 
if uploaded_file is not None: 
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)
    
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)
    
    string_data = stringio.read()
    
    #r = str(uploaded_file.read())
    s = string_data.replace("""b'<?xml version="1.0" encoding="UTF-8"?>\\n""", """""")
    t = s.replace("""\\n""", """""")
    #st.write(type(t))
    #st.write(t)
    f, t, ac = parse(t) 
    st.write(ac.aircraft_type, t[0])
    #st.write(f.dataframe)
    
    #map 
    m = map_route(f.dataframe)
    st_data = st_folium(m, width=725)
    #st.write(type(m))
      
    
    if st.button('Run hello'):
    #if st.checkbox('Compute Model'):
        st.write("Getting Met data")
        era5pl = ERA5(
           time=t,
           variables=Cocip.met_variables + Cocip.optional_met_variables,
           pressure_levels=pressure_levels,
           url='https://cds.climate.copernicus.eu/api/v2', key='204313:423417aa-a534-4646-9728-a7f37618da8c'
        )
        era5sl = ERA5(time=t, variables=Cocip.rad_variables, url='https://cds.climate.copernicus.eu/api/v2', key='204313:423417aa-a534-4646-9728-a7f37618da8c')

        met = era5pl.open_metdataset()
        rad = era5sl.open_metdataset()
        st.write("Done")
        st.write("Evaluating ..")
        
        cocip = Cocip(met=met, rad=rad, params=params)
        #flight = flight.resample_and_fill(freq='5T', drop=False)
        fl_out = cocip.eval(source=f)
        flight_statistics = cocip.output_flight_statistics()
        st.write("Done~!")

        total_contrail_EF = flight_statistics['Total contrail EF (J)']
        st.write(total_contrail_EF) 

