# -*- coding: utf-8 -*-
"""
Eddy Covariance data plotting
@author: David Trejo Cancino
"""

import dash
from dash import html, dcc, register_page, callback
from dash.dependencies import Input, Output
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pathlib
# import matplotlib.pyplot as plt
register_page(__name__, name='OP Eddy Covariance', title='OP EC', description='Peatland Eddy Covariance Visualization')

# %% Data reading
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
# PATH = 'E:/Omora data/Peatland/Eddy/T220819/Eddy20221013/output/221020/'
FILENAME = 'eddypro_omora-peatland_full_output_2022-10-20T151315_adv.csv'
df = pd.read_csv(DATA_PATH.joinpath(FILENAME), skiprows=[0], low_memory=False)
# df = pd.read_csv(PATH+FILENAME, skiprows=[0], low_memory=False)
cols = df.columns.to_list()
units = df.iloc[0]; df = df[1:]
df.index = pd.DatetimeIndex(df['date'].astype(str) + ' '+ df['time'].astype(str))
df = df.astype(float, errors="ignore")
#%%
FILENAME = 'eddypro_omora-peatland_biomet_2022-10-20T083043_adv2.csv'
df2 = pd.read_csv(DATA_PATH.joinpath(FILENAME))
cols2 = df2.columns.to_list(); units2 = df2.iloc[0]; df2 = df2[1:]
df2.index = pd.DatetimeIndex(df2['date'].astype(str) + ' '+ df2['time'].astype(str))
#%% functions
def var_reading(df, units, col):
    try:
        var = np.float64(df[col])
        var[var<=-9999] = np.nan
    except ValueError:
        var = df[col]
    # return (var, units[col])
    return var

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

#%%
# plt.plot(var_reading(df, units, 'h2o_mean.1'), label='h2o_mean.1')
# plt.plot(var_reading(df, units, 'co2_mean.1'), label='co2_mean.1')
# plt.plot(var_reading(df, units, 'h2o_mole_fraction'), label='h2o_mole_fraction')
# plt.plot(var_reading(df, units, 'co2_mixing_ratio'), label='co2_mixing_ratio')



#%% Variables
spec = [[{"secondary_y": True} for i in range(1)] for j in range(4)]
date = df.index
# Diagnostics
battery = var_reading(df2, units2, 'VIN_1_1_1')
# gas and fluxes
co2 = var_reading(df, units, 'co2_mean.1')
h2o = var_reading(df, units, 'h2o_mean.1')
fco2 = var_reading(df, units, 'co2_flux')
fh2o = var_reading(df, units,'h2o_flux')
ch4 = var_reading(df, units, 'ch4_mean'); ch4[ch4<1.8] = np.nan
fch4 = var_reading(df, units, 'ch4_flux')
CO2sig = var_reading(df, units, 'co2_signal_strength_7200_mean')
H2Osig = var_reading(df, units, 'h2o_signal_strength_7200_mean')
# Meteo full_output
rh = var_reading(df, units, 'RH')
vpd = var_reading(df, units, 'VPD')
et = var_reading(df, units, 'ET')/ 2  # 1hour agg to 30 min agg
ta = var_reading(df, units, 'air_temperature')
Td = var_reading(df, units, 'Tdew')
p = var_reading(df, units, 'air_pressure')/1000
# Wind
wind = var_reading(df, units, 'wind_speed')
u = var_reading(df, units, 'u_mean')
v = var_reading(df, units, 'v_mean')
w = var_reading(df, units, 'w_mean')
wind_dir = var_reading(df, units, 'wind_dir')
u_star = var_reading(df, units, 'u*')
tau = var_reading(df, units, 'Tau')
# Energy
Hc = var_reading(df, units, 'H')
LE = var_reading(df, units, 'LE')
TCNR4 = var_reading(df2, units2, 'TCNR4_C_1_1_1') # Con error
SW = var_reading(df2, units2, 'SWIN_1_1_1') - var_reading(df2, units2, 'SWOUT_1_1_1')
LWin = var_reading(df2, units2, 'LWIN_1_1_1')
LWout = var_reading(df2, units2, 'LWOUT_1_1_1')
def LW_body_temp_correction(LW, NR01TK):
    delta = 5.67 * 10**(-8)
    e = LW + (delta * (NR01TK)**(4))
    return e
LWin = LW_body_temp_correction(LWin, ta)
LWout = LW_body_temp_correction(LWout, ta)
LW = LWin - LWout
Rn = var_reading(df2, units, 'RN_1_1_1')
NETRAD = var_reading(df2, units, 'RN_1_1_1')
SHF_COMPONENTS = df2.filter(like='SHF_').astype(float)
SHF_COMPONENTS[SHF_COMPONENTS<-999] = np.nan
SHF = SHF_COMPONENTS.mean(axis=1)
CLOSURE = np.float64(Rn) - np.float64(SHF) - np.float64(LE) - np.float64(Hc)
PPFD = var_reading(df2, units, 'PPFD_1_1_1')
ALB = var_reading(df2, units, 'ALB_1_1_1')
# Fluxes and concentrations

# Met
pp = var_reading(df2, units, 'P_RAIN_1_1_1')
RH = var_reading(df2, units, 'RH_1_1_1')
SWC_COMPONENTS = df2.filter(like='SWC_').astype(float)
SWC_COMPONENTS[SWC_COMPONENTS<-1] = np.nan
Tsoil_COMPONENTS = df2.filter(like='TS_').astype(float)
Tsoil_COMPONENTS[Tsoil_COMPONENTS<-999] = np.nan
# WTD_COMPONENTS = df_fst.filter(like='Level_')
# WTD_max = WTD_COMPONENTS.filter(like='_Max')
# WTD_min = WTD_COMPONENTS.filter(like='_Min')

# Meteorological
del df
# %% function filtering the wind rose data based on the wind speed and direction (NEEDED BELOW)
def wind_dir_speed_freq(boundary_lower_speed, boundary_higher_speed, boundary_lower_direction,
                        boundary_higher_direction):
    # mask for wind speed column
    log_mask_speed = (wind_rose_data[:, 0] >= boundary_lower_speed) & (wind_rose_data[:, 0] < boundary_higher_speed)
    # mask for wind direction
    log_mask_direction = (wind_rose_data[:, 1] >= boundary_lower_direction) & (
                wind_rose_data[:, 1] < boundary_higher_direction)

    # application of the filter on the wind_rose_data array
    return wind_rose_data[log_mask_speed & log_mask_direction]


# %% Wind rose

# Creating a pandas dataframe with 8 wind speed bins for each of the 16 wind directions.
# dataframe structure: direction | strength | frequency (radius)

wind_rose_df_fst = pd.DataFrame(np.zeros((16 * 9, 3)), index=None, columns=('direction', 'strength', 'frequency'))

directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
directions_deg = np.array([0, 22.5, 45, 72.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5])
speed_bins = ['0-2 m/s', '2-4 m/s', '4-6 m/s', '6-8 m/s', '8-10 m/s', '10-12 m/s', '12-14 m/s', '14-25 m/s', '>25 m/s']

# filling in the dataframe with directions and speed bins
wind_rose_df_fst.direction = directions * 9
wind_rose_df_fst.strength = np.repeat(speed_bins, 16)

# creating a multiindex dataframe with frequencies

idx = pd.MultiIndex.from_product([speed_bins,
                                  directions_deg],
                                 names=['wind_speed_bins', 'wind_direction_bins'])
col = ['frequency']
frequencies_df_fst = pd.DataFrame(0, idx, col)

wind_rose_data = np.asarray([np.float64(wind), np.float64(wind_dir)]).T
# meteo[['wind_speed_m/s', 'wind_direction_deg']].to_numpy()

# distance between the centre of the bin and its edge
step = 11.25

# converting data between 348.75 and 360 to negative
for i in range(len(wind_rose_data)):
    if directions_deg[-1] + step <= wind_rose_data[i, 1] and wind_rose_data[i, 1] < 360:
        wind_rose_data[i, 1] = wind_rose_data[i, 1] - 360

# determining the direction bins
bin_edges_dir = directions_deg - step
bin_edges_dir = np.append(bin_edges_dir, [directions_deg[-1] + step])

# determining speed bins ( the last bin is 50 as above those speeds the outliers were removed for the measurements)
threshold_outlier_rm = 50
bin_edges_speed = np.array([0, 2, 4, 6, 8, 10, 12, 14, 25, threshold_outlier_rm])

frequencies = np.array([])
# loop selecting given bins and calculating frequencies
for i in range(len(bin_edges_speed) - 1):
    for j in range(len(bin_edges_dir) - 1):
        bin_contents = wind_dir_speed_freq(bin_edges_speed[i], bin_edges_speed[i + 1], bin_edges_dir[j],
                                           bin_edges_dir[j + 1])

        # applying the filtering function for every bin and checking the number of measurements
        bin_size = len(bin_contents)
        frequency = bin_size / len(wind_rose_data)

        # obtaining the final frequencies of bin
        frequencies = np.append(frequencies, frequency)

# updating the frequencies dataframe
frequencies_df_fst.frequency = frequencies * 100  # [%]
wind_rose_df_fst.frequency = frequencies * 100  # [%]
# %% web app
fig_names = ['Battery', 'GHG Concentrations', 'Fluxes', 'Wind',
             'Wind Direction', 'Energy Balance', 'Biomet', 'Multi-channel signals']

layout = html.Div([
    html.H1('Eddy Data from Peatland Omora, CHIC',
            style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Pre(children="Variables", style={"fontSize": "150%"}),
            dcc.Dropdown(
                id='fig-dropdown', value='GHG Concentrations', clearable=False,
                persistence=True, persistence_type='session',
                options=[{'label': x, 'value': x} for x in fig_names])],
            className='six columns'), ], className='row'),

    dcc.Graph(id='my-map', figure={}),
])


@callback(
    Output(component_id='my-map', component_property='figure'),
    [Input(component_id='fig-dropdown', component_property='value')])



def name_to_figure(fig_name):
    if fig_name == 'Battery':
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(go.Scatter(x=date, y=battery, mode='lines', name='Battery Voltage'),
                      row=1, col=1)
        fig.update_yaxes(range=[10, 13], title_text="Voltage Input",
                         row=1, col=1)
        fig.update_layout(autosize=False, width=1750, height=500)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True,
            xaxis_rangeslider_visible=True, xaxis_type="date",
            xaxis_rangeslider_bgcolor='grey', xaxis_rangeslider_thickness=0.03)
    elif fig_name == 'GHG Concentrations':
        fig = make_subplots(rows=3, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}],
                                                   [{"secondary_y": True}]],
                            shared_xaxes=True)
        # CO2
        fig.add_trace(go.Scatter(x=date, y=co2, mode='lines', name='CO2 mole fraction',
                                 line=dict(color='blue')), row=1, col=1)
        fig.update_yaxes(range=[320, 550], title_text="CO2 [μmol mol-1]",
                         row=1, col=1, titlefont=dict(color="blue"),
                         tickfont=dict(color="blue"))
        fig.add_trace(go.Scatter(x=date, y=CO2sig, mode='lines', name='CO2 signal strength',
                                 line=dict(color='red')), secondary_y=True, row=1, col=1)
        fig.update_yaxes(range=[20, 120], title_text="CO2 signal strength",
                         row=1, col=1, secondary_y=True, titlefont=dict(color="red"),
                         tickfont=dict(color="red"))
        # H2O
        fig.add_trace(go.Scatter(x=date, y=h2o, mode='lines', name='H2O mole fraction',
                                 line=dict(color='green')),
                      row=2, col=1)
        fig.update_yaxes(range=[1, 20], title_text="H2O [mmol mol-1]",
                         row=2, col=1, titlefont=dict(color="green"),
                         tickfont=dict(color="green"))
        fig.add_trace(go.Scatter(x=date, y=H2Osig, mode='lines', name='H2O signal strength',
                                 line=dict(color='purple')),
                      row=2, col=1, secondary_y=True)
        fig.update_yaxes(range=[20, 120], title_text="H2O signal strength",
                         row=2, col=1, secondary_y=True, titlefont=dict(color="purple"),
                         tickfont=dict(color="purple"))
        # CH4
        fig.add_trace(go.Scatter(x=date, y=ch4, mode='lines', name='CH4 mole fraction',
                                 line=dict(color='blue')), row=3, col=1)
        fig.update_yaxes(range=[1.7, 2], title_text="CH4 [μmol mol-1]",
                         row=3, col=1, titlefont=dict(color="blue"),
                         tickfont=dict(color="blue"))
        # fig.update_layout(autosize=False, width=1750, height=500)
        fig.update_layout(autosize=False, width=1750, height=1300)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True,
            xaxis2_rangeslider_visible=True, xaxis2_type="date",
            xaxis2_rangeslider_bgcolor='grey', xaxis2_rangeslider_thickness=0.03)
    elif fig_name == 'Fluxes':
        fig = make_subplots(rows=5, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}], [{"secondary_y": True}],
                                                   [{"secondary_y": True}],
                                                   [{"secondary_y": True}]],
                            shared_xaxes=True)
        # CO2
        fig.add_trace(go.Scatter(x=date, y=fco2, mode='lines', name='CO2 Flux',
                                 line=dict(color='blue')), row=1, col=1)
        fig.update_yaxes(range=[-8, 8], title_text="CO2 Flux [μmol m-2 s-1]",
                         row=1, col=1, titlefont=dict(color="blue"),
                         tickfont=dict(color="blue"))
        fig.add_trace(go.Scatter(x=date, y=CO2sig, mode='lines', name='CO2 signal strength',
                                 line=dict(color='red')), secondary_y=True, row=1, col=1)
        fig.update_yaxes(range=[20, 110], title_text="CO2 signal strength",
                         row=1, col=1, secondary_y=True, titlefont=dict(color="red"),
                         tickfont=dict(color="red"))
        # H2O
        fig.add_trace(go.Scatter(x=date, y=fh2o, mode='lines', name='H2O Flux',
                                 line=dict(color='green')),
                      row=2, col=1)
        fig.update_yaxes(range=[-0.8, 0.8], title_text="H2O Flux [mmol m-2 s-1]",
                         row=2, col=1, titlefont=dict(color="green"),
                         tickfont=dict(color="green"))
        fig.add_trace(go.Scatter(x=date, y=H2Osig, mode='lines', name='H2O signal strength',
                                 line=dict(color='purple')),
                      row=2, col=1, secondary_y=True)
        fig.update_yaxes(range=[20, 110], title_text="H2O signal strength",
                         row=2, col=1, secondary_y=True, titlefont=dict(color="purple"),
                         tickfont=dict(color="purple"))
        # CH4
        fig.add_trace(go.Scatter(x=date, y=fch4, mode='lines', name='Methane Flux',
                                 line=dict(color='firebrick')),
                      row=3, col=1)
        fig.update_yaxes(range=[-0.3, 0.3], title_text="CH4 Flux [μmol m-2 s-1]",
                         row=3, col=1, titlefont=dict(color="firebrick"),
                         tickfont=dict(color="firebrick"))
        # ET
        fig.add_trace(go.Scatter(x=date, y=et, mode='lines', name='Evapotranspiration (ET)'),
                      row=4, col=1)
        fig.update_yaxes(range=[-0.1, 0.1], title_text="ET [mm 30min-1]",
                         row=4, col=1)
        # Momentum flux
        fig.add_trace(go.Scatter(x=date, y=tau, mode='lines', name='Momentum Flux (Tau)'),
                      row=5, col=1)
        fig.update_yaxes(range=[-0.8, 0.8], title_text="Tau [kg s-2 m-1]",
                         row=5, col=1)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True,
            xaxis3_showticklabels=True, xaxis4_showticklabels=True,
            xaxis5_showticklabels=True,
            xaxis5_rangeslider_visible=True, xaxis5_type="date",
            xaxis5_rangeslider_bgcolor='grey', xaxis5_rangeslider_thickness=0.03)
        fig.update_layout(autosize=False, width=1750, height=1400)
    elif fig_name == 'Wind':
        fig = make_subplots(rows=3, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}],
                                                   [{"secondary_y": True}]])
        
        # Wind Speed and friction velocity
        fig.add_trace(go.Scatter(x=date, y=wind, mode='lines',
                                 name='Wind Speed', line=dict(color='grey')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=u_star, mode='lines',
                                 name='Friction Velocity', line=dict(color='orange')),
                      row=1, col=1, secondary_y=True)
        fig.update_layout(autosize=False, width=1750, height=1500)
        fig.update_yaxes(range=[0, 10], title_text="Wind Speed [m/s]",
                         row=1, col=1, titlefont=dict(color="grey"),
                         tickfont=dict(color="teal"))
        fig.update_yaxes(range=[0, 2], title_text="Friction Velocity [m/s]",
                         row=1, col=1, titlefont=dict(color="orange"),
                         tickfont=dict(color="orange"), secondary_y=True)
        fig.update_xaxes(row=1, col=1, matches='x')
        # Wind Direction Series
        fig.add_trace(go.Bar(x=date, y=wind_dir,
                             name='Wind Direction Time Series', marker_color='forestgreen'),
                      row=2, col=1)
        fig.update_yaxes(range=[-1, 361],
                         title_text="Degrees from North", row=2, col=1)
        fig.update_xaxes(row=2, col=1, matches='x')
        # Wind direction Histogram
        fig.add_trace(go.Histogram(x=wind_dir, histnorm='probability',
                                   marker_color='#330C73', name='Wind Direction Histogram'),
                      row=3, col=1)
        fig.update_yaxes(title_text="Normalized Probability", row=3, col=1)
        fig.update_xaxes(title_text="Degrees from North", row=3, col=1)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis2_rangeslider_visible=True, xaxis2_type="date",
            xaxis2_rangeslider_bgcolor='grey', xaxis2_rangeslider_thickness=0.03)
        fig.update_layout(autosize=False, width=1750, height=1200)
    elif fig_name == 'Wind Direction':
        fig = go.Figure()
        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('0-2 m/s'), 'frequency'],
            name='0-2 m/s',
            marker_color='#482878'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('2-4 m/s'), 'frequency'],
            name='2-4 m/s',
            marker_color='#3e4989'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('4-6 m/s'), 'frequency'],
            name='4-6 m/s',
            marker_color='#31688e'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('6-8 m/s'), 'frequency'],
            name='6-8 m/s',
            marker_color='#26828e'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('8-10 m/s'), 'frequency'],
            name='8-10 m/s',
            marker_color='#1f9e89'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('10-12 m/s'), 'frequency'],
            name='10-12 m/s',
            marker_color='#35b779'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('12-14 m/s'), 'frequency'],
            name='12-14 m/s',
            marker_color='#6ece58'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('14-25 m/s'), 'frequency'],
            name='14-25 m/s',
            marker_color='#b5de2b'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('>25 m/s'), 'frequency'],
            name='>25 m/s',
            marker_color='#fde725'))

        fig.update_traces(text=['North', 'NNE', 'NE', 'ENE', 'East', 'ESE',
                                'SE', 'SSE', 'South', 'SSW', 'SW', 'WSW', 'West', 'WNW', 'NW', 'NNW'])

        fig.update_layout(
            autosize=False, width=1500, height=800,
            title='Wind Rose',
            title_font_size=26,
            title_x=0.463,
            legend_font_size=18,
            polar_radialaxis_ticksuffix='%',
            polar_angularaxis_rotation=90,
            polar_angularaxis_direction='clockwise',
            polar_angularaxis_tickmode='array',
            polar_angularaxis_tickvals=[0, 22.5, 45, 72.5, 90, 112.5, 135,
                                        157.5, 180, 202.5, 225, 247.5, 270,
                                        292.5, 315, 337.5],
            polar_angularaxis_ticktext=['<b>North</b>', 'NNE', '<b>NE</b>',
                                        'ENE', '<b>East</b>', 'ESE', '<b>SE</b>',
                                        'SSE', '<b>South</b>', 'SSW', '<b>SW</b>',
                                        'WSW', '<b>West</b>', 'WNW', '<b>NW</b>',
                                        'NNW'],
            polar_angularaxis_tickfont_size=22,
            polar_radialaxis_tickmode='linear',
            polar_radialaxis_angle=45,
            polar_radialaxis_tick0=5,
            polar_radialaxis_dtick=5,
            polar_radialaxis_tickangle=100,
            polar_radialaxis_tickfont_size=14)

    elif fig_name == 'Energy Balance':
        # 30 min data of energy components
        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}]], shared_xaxes=True)
        fig.add_trace(go.Scatter(x=date, y=SW, mode='lines',
                                  name='Net Short Wave Radiation', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=LW, mode='lines',
                                  name='Net Long Wave Radiation', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=NETRAD, mode='lines',
                                  name='NET Radiation from NR Lite', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=Rn, mode='lines',
                                  name='Net Radiation from NR01 (4 components)', line=dict(color='green')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=LE, mode='lines',
                                 name='Latent Heat', line=dict(color='purple')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=Hc, mode='lines',
                                 name='Sensible Heat', line=dict(color='brown')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=SHF, mode='lines',
                                  name='Soil Heat Flux', line=dict(color='pink')), row=1, col=1)
        fig.update_yaxes(range=[-500, 1300], title_text="Energy Components [W/m2]",
                         row=1, col=1)
        #  30 min data of energy balance Closure
        fig.add_trace(go.Scatter(x=date, y=CLOSURE, mode='lines', name='CLOSURE',
                                  line=dict(color='gray')), row=2, col=1)
        fig.update_yaxes(range=[-700, 700], title_text="Closure NetRad - SHF - LE - H [W/m2]",
                          row=2, col=1)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True,
            xaxis2_rangeslider_visible=True, xaxis2_type="date",
            xaxis2_rangeslider_bgcolor='grey', xaxis2_rangeslider_thickness=0.03)
        fig.update_layout(autosize=False, width=1750, height=1000)
    elif fig_name == 'Biomet':
        # 30 min data of energy components
        fig = make_subplots(rows=4, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}], [{"secondary_y": True}],
                                                   [{"secondary_y": True}]],
                            shared_xaxes=True)
        # Temperature
        fig.add_trace(go.Scatter(x=date, y=ta-273.15, mode='lines', name='Ambient Temperature',
                                 line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=Td-273.15, mode='lines', name='Dew Point',
                                 line=dict(color='orange')), row=1, col=1)
        fig.update_yaxes(range=[-10, 32], title_text="°C",
                         row=1, col=1)
        # Humidity
        fig.add_trace(go.Scatter(x=date, y=rh, mode='lines', name='Relative Humidity IRGA',
                                 line=dict(color='blue')), row=2, col=1)
        # fig.add_trace(go.Scatter(x=date, y=r, mode='lines', name='H2O Mixing Ratio HMP',
                                 # line=dict(color='orange')), row=2, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=date, y=RH, mode='lines', name='Relative Humidity HMP',
                                  line=dict(color='peru')), row=2, col=1, secondary_y=True)
        fig.update_yaxes(range=[15, 105], title_text="[%]",
                         row=2, col=1, titlefont=dict(color="blue"),
                         tickfont=dict(color="blue"))
        # Pressure
        fig.add_trace(go.Scatter(x=date, y=p, mode='lines', name='Ambient Pressure'),
                      row=3, col=1)
        fig.update_yaxes(range=[95, 110], title_text="[kPa]",
                         row=3, col=1)
        # Precipitation
        fig.add_trace(go.Scatter(x=date, y=pp, mode='lines', name='Precipitation'),
                      row=4, col=1)
        fig.update_yaxes(range=[0, 0.1], title_text="[mm]",
                          row=4, col=1)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True, xaxis3_showticklabels=True,
            xaxis4_showticklabels=True,
            xaxis4_rangeslider_visible=True, xaxis4_type="date",
            xaxis4_rangeslider_bgcolor='grey', xaxis4_rangeslider_thickness=0.03)
        fig.update_layout(autosize=False, width=1750, height=1500)
    elif fig_name == 'Multi-channel signals':
        fig = make_subplots(rows=3, cols=1)
        for i in range(np.shape(SHF_COMPONENTS)[1]):
            shf = SHF_COMPONENTS.iloc[:, i]
            fig.add_trace(go.Scatter(x=date, y=shf, mode='lines',
                                      name='SHF ' + str(i + 1)), row=1, col=1)
            fig.update_yaxes(range=[-30, 30], title_text="Soil Heat Flux [W m-2]",
                              row=1, col=1)
        for i in range(np.shape(SWC_COMPONENTS)[1]):
            swc = SWC_COMPONENTS.iloc[:, i]
            fig.add_trace(go.Scatter(x=date, y=swc, mode='lines',
                                      name='SWC ' + str(i + 1)), row=2, col=1)
            fig.update_yaxes(range=[0.4, 0.45], title_text="Soil Water Content [m3 m-3]",
                              row=2, col=1)
        for i in range(np.shape(Tsoil_COMPONENTS)[1]):
            Tsoil = Tsoil_COMPONENTS.iloc[:, i]
            fig.add_trace(go.Scatter(x=date, y=Tsoil-273.15, mode='lines',
                                      name='Tsoil ' + str(i + 1)), row=3, col=1)
            fig.update_yaxes(range=[-2, 10], title_text="Soil Temperature [°C]",
                              row=3, col=1)

        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                              label="1m",
                              step="month",
                              stepmode="backward"),
                        dict(count=6,
                              label="6m",
                              step="month",
                              stepmode="backward"),
                        dict(count=1,
                              label="YTD",
                              step="year",
                              stepmode="todate"),
                        dict(count=1,
                              label="1y",
                              step="year",
                              stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True, xaxis3_showticklabels=True,
            xaxis3_rangeslider_visible=True, xaxis3_type="date",
            xaxis3_rangeslider_bgcolor='grey', xaxis3_rangeslider_thickness=0.03)
        fig.update_layout(autosize=False, width=1750, height=1500)
    
    return fig  # dcc.Graph(fig=fig)