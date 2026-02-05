

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

import random 
from torch.utils.data import Dataset, Subset
import collections
from common.main import configs, data, trainloader
from common.utils import image_to_base64 
    
def show_images(loader):
    # Get one batch
    images, labels = next(iter(loader))
    encoded_images = [image_to_base64(img) for img in images]
    # Create a list of image components with labels
    image_grid = [
        html.Div([
            html.Img(src=f'data:image/png;base64,{img}', style={"height": "100px"}),
            html.Div(f"Label: {labels[i].item()}", style={"text-align": "center"})
        ], style={"margin": "10px", "text-align": "center"})
        for i, img in enumerate(encoded_images)
    ] 
    return image_grid       

def create_label_distribution_figure(dataset: Dataset | Subset) -> go.Figure:
    labels = [label for _, label in dataset]
    counter = collections.Counter(labels)

    keys = list(counter.keys())
    values = list(counter.values())

    fig = go.Figure(
        data=[go.Bar(x=keys, y=values, marker_color='skyblue')],
        layout=go.Layout(
            title="Label Count",
            xaxis_title="Labels",
            yaxis_title="Amount",
            template="plotly_dark"
        )
    )
    return fig       

def create_overview_table():
    row1 = html.Tr([html.Td("Name"), html.Td(data.name)])
    row2 = html.Tr([html.Td("Path"), html.Td(data.path)])
    row3 = html.Tr([html.Td("Training Size"), html.Td(len(data.train_dataset))])
    row4 = html.Tr([html.Td("Test Size"), html.Td(len(data.test_dataset))])

    table_body = [html.Tbody([row1, row2, row3, row4])]

    return dbc.Table(
        table_body,
        bordered=True,
        color="secondary",
        hover=True,
        responsive=True,
        striped=True,
    )                                                                                         

dash.register_page(__name__, path='/data')

layout = html.Div([
    html.H3('Overview: Data'),
    create_overview_table(),
    html.Hr(),
    html.H3('Examples'),
    html.Div(
        children=show_images(trainloader),
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "5px",
            "justifyItems": "center",
            "width": "50%",
        }
    ),
    html.Hr(),
    html.H3('Distributions'),
    html.Div([
        html.Div([
            html.H4("Train Set"),
            dcc.Graph(figure=create_label_distribution_figure(data.train_dataset))
        ], style={"width": "50%", "padding": "10px"}),

        html.Div([
            html.H4("Test Set"),
            dcc.Graph(figure=create_label_distribution_figure(data.test_dataset))
        ], style={"width": "50%", "padding": "10px"}),
    ], style={"display": "flex", "flexWrap": "wrap"}),
    html.Hr(),
    html.H3('Select Image'),
    html.Div(
        [dcc.Input(
        id="index-input",
        type="number",
        step=1,
        min=0,
        max=999,
        value=0,
        placeholder="Enter index",
        style={
            "marginBottom": "20px", "marginRight": "20px", 
            "lineHeight": "38px", 
            "width": "25vw",
            "height": "38px"      
        }
        ),
        dcc.Dropdown(
            id="set-dropdown", 
            options=["train", "test"], 
            placeholder="Choose a set", 
            value="train", style = {
            "lineHeight": "38px", 
            "width": "25vw",
            "height": "38px"  
        })
        ], style={"display": "flex", "flexWrap": "wrap"}),
    html.Div(id="output", style={"marginBottom": "20px"}),
    html.Hr(),
])

@callback(
    Output("output", "children"),
    Input("index-input", "value"),
    Input("set-dropdown", "value")
)
def show_input_value(value, set_value):
    if set_value == "train":
        image, label = data.train_dataset[value]
    else:
        image, label = data.test_dataset[value]
    image = image_to_base64(image)
    return html.Img(src=f'data:image/png;base64,{image}', style={"height": "100px"}),