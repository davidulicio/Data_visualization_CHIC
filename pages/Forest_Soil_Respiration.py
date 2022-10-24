# -*- coding: utf-8 -*-
"""
Soil Respiration data plotting
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

register_page(__name__, name="OF Soil Respiration", title='OF SR', description='Forest Soil Respiration Visualization')

# %% Data reading
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
FILENAME = 'OF221024.csv'
df = pd.read_csv(DATA_PATH.joinpath(FILENAME), skiprows=[0])
cols = df.columns.to_list()
units = df.iloc[0]; df = df[1:]
#%% functions
def var_reading(df, units, col):
    try:
        var = np.float64(df[col])
        # var[var<=-9999] = np.nan
    except ValueError:
        var = df[col]
    return (var, units[col])

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
          "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
#%% Variables
date = pd.DatetimeIndex(df['DATE_TIME initial_value'])
# Identifiers
cham_id = var_reading(df, units, 'PORT_LABEL')[0]
# cham_sn = var_reading(df, units, 'SERIAL_NUMBER')[0]
mask = [i==cham_id for i in range(1, 5)] 
spec = [[{"secondary_y": True} for i in range(1)] for j in range(4)]
# Diagnostics
diag_val = var_reading(df, units, 'DIAG mean')
flow_870 = var_reading(df, units, 'FLOW mean')
flow_8250 = var_reading(df, units, 'FLOW mean.1')
voltage_870 = var_reading(df, units, 'VIN mean')
T_case = var_reading(df, units, 'T_CASE mean')  # 8250
T_cell = var_reading(df, units, 'T_CELL mean')  #870
Pa_cell = var_reading(df, units, 'PA_CELL mean')  #870
# gas and fluxes
co2 = var_reading(df, units, 'CO2 mean')
dry_co2 = var_reading(df, units,'CO2_DRY mean')
h2o = var_reading(df, units, 'H2O mean')
abs_co2 = var_reading(df, units, 'CO2_ABS mean')
abs_h2o = var_reading(df, units, 'H2O_ABS mean')
exp_fco2 = var_reading(df, units, 'FCO2_DRY')
exp_co2_r2 = var_reading(df, units, 'FCO2_DRY R2')
exp_co2_cv = var_reading(df, units, 'FCO2_DRY CV')
exp_fh2o = var_reading(df, units,'FH2O')
exp_h2o_r2 = var_reading(df, units, 'FH2O R2')
exp_h2o_cv = var_reading(df, units, 'FH2O CV')
lin_fco2 = var_reading(df, units,'FCO2_DRY LIN')
lin_co2_r2 = var_reading(df, units, 'FCO2_DRY LIN_R2')
lin_co2_cv = var_reading(df, units, 'FCO2_DRY LIN_CV')
lin_fh2o = var_reading(df, units, 'FH2O LIN')
lin_h2o_r2 =  var_reading(df, units, 'FH2O LIN_R2')
lin_h2o_cv =  var_reading(df, units,'FH2O LIN_CV')
# Meteorological
# %% web app
fig_names = ['Diagnostics', 'Gases', 'Fluxes']

layout = html.Div([
    html.H1('Soil Respiration Data from Forest Omora, CHIC',
            style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Pre(children="Variables", style={"fontSize": "150%"}),
            dcc.Dropdown(
                id='fig-dropdown3', value='Battery', clearable=False,
                persistence=True, persistence_type='session',
                options=[{'label': x, 'value': x} for x in fig_names])],
            className='six columns'), ], className='row'),

    dcc.Graph(id='my-map3', figure={}),
])

@callback(
    Output(component_id='my-map3', component_property='figure'),
    [Input(component_id='fig-dropdown3', component_property='value')])



def name_to_figure(fig_name):
    if fig_name == 'Diagnostics':
        fig = make_subplots(rows=3, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": False}],
                                                   [{"secondary_y": True}]],
                            shared_xaxes=True)
        
        # Diagnostics and voltage
        fig.add_trace(go.Scatter(x=date, y=voltage_870[0], mode='lines', name='Voltage LI-870',
                                 line=dict(color='deepskyblue')), row=1, col=1)
        fig.update_yaxes(range=[8, 14], title_text="Voltage",
                         row=1, col=1, titlefont=dict(color="blue"),
                         tickfont=dict(color="blue"))
        fig.add_trace(go.Scatter(x=date, y=diag_val[0], mode='lines', name='Diagnostics value',
                                 line=dict(color='red')), secondary_y=True, row=1, col=1)
        fig.update_yaxes(range=[-0.1, 1.1], title_text="Diagnostics [#]",
                         row=1, col=1, secondary_y=True, titlefont=dict(color="red"),
                         tickfont=dict(color="red"))
        ####
        # flowrate
        fig.add_trace(go.Scatter(x=date, y=flow_8250[0], mode='lines', name='Flowrate LI-8250',
                                 line=dict(color='goldenrod')), row=2, col=1)
        fig.add_trace(go.Scatter(x=date, y=flow_870[0], mode='lines', name='Flowrate LI-870',
                                 line=dict(color='green')), row=2, col=1)
        fig.update_yaxes(range=[0, 4], title_text="Flowrate [L min-1]",
                         row=2, col=1, titlefont=dict(color="grey"))
        # Case and cell values
        fig.add_trace(go.Scatter(x=date, y=T_case[0], mode='lines', name='Case Temperature (LI-8250)',
                                 line=dict(color='indigo')), row=3, col=1)
        fig.add_trace(go.Scatter(x=date, y=T_cell[0], mode='lines', name='Cell Temperature (LI-870)',
                                 line=dict(color='darkslateblue')), row=3, col=1)
        fig.update_yaxes(range=[0, 55], title_text="Temperature [°C]",
                         row=3, col=1, titlefont=dict(color="indigo"),
                         tickfont=dict(color="indigo"))
        fig.add_trace(go.Scatter(x=date, y=Pa_cell[0], mode='lines', name='Cell Pressure',
                                 line=dict(color='deeppink')), secondary_y=True, row=3, col=1)
        fig.update_yaxes(range=[90, 120], title_text="Pressure [kPa]",
                         row=3, col=1, secondary_y=True, titlefont=dict(color="deeppink"),
                         tickfont=dict(color="deeppink"))
        fig.update_layout(autosize=False, width=1750, height=800)
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
            xaxis3_rangeslider_visible=True, xaxis3_type="date",
            xaxis3_rangeslider_bgcolor='grey', xaxis3_rangeslider_thickness=0.03)
    elif fig_name == 'Gases':
        
        spec = [[{"secondary_y": True} for i in range(1)] for j in range(3)]
        fig = make_subplots(rows=3, cols=1, specs=spec, shared_xaxes=True)
        for i in range(len(mask)):
            # CO2
            fig.add_trace(go.Scatter(x=date[mask[i]], y=co2[0][mask[i]],
                         mode='lines', name='Mean CO2 of chamber '+ str(i+1)),
                         row=1, col=1)

        fig.add_trace(go.Scatter(x=date, y=abs_co2[0], mode='lines', name='CO2 Absorption',
                                      line=dict(color='lime')), row=1, col=1,
                          secondary_y=True)
        fig.update_yaxes(range=[250, 800], title_text="CO2 " + co2[1],
                          row=1, col=1)
        fig.update_yaxes(range=[0, 0.15], title_text="CO2 Absorption " + abs_co2[1],
                          row=1, col=1, secondary_y=True, titlefont=dict(color="lime"),
                          tickfont=dict(color="lime"))
        # H2O
        for i in range(len(mask)):
            fig.add_trace(go.Scatter(x=date[mask[i]], y=h2o[0][mask[i]],
                         mode='lines', name='Mean H2O of chamber '+ str(i+1)),
                          row=2, col=1)
        fig.update_yaxes(range=[2, 20], title_text="H2O " + h2o[1],
                          row=2, col=1)
        fig.add_trace(go.Scatter(x=date, y=abs_h2o[0], mode='lines', name='H2O Absorption',
                                 line=dict(color='gold')), row=2, col=1,
                      secondary_y=True)
        fig.update_yaxes(range=[0, 0.15], title_text="H2O Absorption " + abs_h2o[1],
                         row=2, col=1, secondary_y=True, titlefont=dict(color="gold"),
                         tickfont=dict(color="gold"))
        fig.update_layout(autosize=False, width=1750, height=800)
    elif fig_name == 'Fluxes':
        spec = [[{"secondary_y": True} for i in range(1)] for j in range(4)]
        fig = make_subplots(rows=4, cols=1, specs=spec, shared_xaxes=True)
        
        # CO2
        for i in range(len(mask)):
            fig.add_trace(go.Scatter(x=date[mask[i]], y=exp_fco2[0][mask[i]],
                         mode='lines', name='Expontential CO2 Flux of chamber '+ str(i+1),
                         line=dict(color=colors[i])), row=1, col=1)
        for i in range(len(mask)):
            fig.add_trace(go.Scatter(x=date[mask[i]], y=exp_co2_r2[0][mask[i]],
                         mode='lines', name='Exponential CO2 R2 of chamber '+ str(i+1),
                         line=dict(color=colors[i])), row=2, col=1)
        fig.update_yaxes(range=[0, 1], title_text="Exponential CO2 R2" + exp_co2_r2[1],
                          row=2, col=1)
        fig.update_yaxes(range=[-1, 5], title_text="Exp CO2 Flux " + exp_fco2[1],
                         row=1, col=1)
        # H2O
        for i in range(len(mask)):
            fig.add_trace(go.Scatter(x=date[mask[i]], y=exp_fh2o[0][mask[i]],
                         mode='lines', name='Expontential H2O Flux of chamber '+ str(i+1),
                         line=dict(color=colors[i+len(mask)])), row=3, col=1)
        for i in range(len(mask)):
            fig.add_trace(go.Scatter(x=date[mask[i]], y=exp_h2o_r2[0][mask[i]],
                         mode='lines', name='Exponential H2O R2 of chamber '+ str(i+1),
                         line=dict(color=colors[i+len(mask)])), row=4, col=1)
        fig.update_yaxes(range=[0, 1], title_text="Exponential H2O R2" + exp_h2o_r2[1],
                          row=3, col=1)
        fig.update_yaxes(range=[-.1, 1.1], title_text="Exp H2O Flux " + exp_fh2o[1],
                         row=4, col=1)
        fig.update_layout(autosize=False, width=1750, height=1200)
    return fig