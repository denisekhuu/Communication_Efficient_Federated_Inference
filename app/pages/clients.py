import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from common.main import number_of_clients, transformed_data, op_transform, grouped_sets, clients
from common.utils import image_to_base64 
import random

dash.register_page(__name__)

def create_overview_table():
    row1 = html.Tr([html.Td("Number of Clients"), html.Td(number_of_clients)])
    row2 = html.Tr([html.Td("Coverage (in %)"), html.Td(op_transform.coverage_percent)])

    table_body = [html.Tbody([row1, row2])]

    return dbc.Table(
        table_body,
        bordered=True,
        color="secondary",
        hover=True,
        responsive=True,
        striped=True,
    )     

def create_tabs_for_dataset():
    label_to_indices = grouped_sets

    tabs = []
    for label, indices in label_to_indices.items():
        random_index = random.randint(0, len(indices) - 1)
        images = [image_to_base64(img.evidence) for img in transformed_data.train_dataset[indices[0][random_index]][0]]
        image_divs = [
            html.Div([
                html.Img(src=f"data:image/png;base64,{img}", style={"height": "100px"}),
                html.Div(f"Client: {i}", style={"textAlign": "center"})
            ], style={"margin": "10px", "textAlign": "center"})
            for i, img in zip(range(len(images)), images)
        ]

        # To change grid of example
        tab_content = html.Div(
            image_divs,
                    style={
            "display": "grid",
            "gridTemplateColumns": "repeat(2, 1fr)",
            "gap": "2px",
            "justifyItems": "center",
            "width": "25%",
        }
        )
        
        tabs.append(dcc.Tab(label=f"Label {label}", children=[tab_content]))

    return dcc.Tabs(tabs)

def generate_client_card(client):
    img = image_to_base64(client.trainset[0][0])
    return dbc.Card(
        [
            dbc.CardImg(src=f"data:image/png;base64,{img}", top=True),
            dbc.CardBody(
                [
                    html.H4(f"Client {client.id}", className="card-title"),
                    html.P("View this client's data.", className="card-text"),
                    dbc.Button("View Client", color="primary", href=f"/client?id={client.id}")
                ]
            ),
        ],
        style={"width": "18rem", "margin": "10px"},
    )

layout = html.Div([
    html.H3("Overview: Client Plane Data"),
    create_overview_table(), 
    html.Hr(),
    html.H3("Example"),
    create_tabs_for_dataset(),
    html.Hr(),
    html.H3("Clients"),
    dbc.Row(
        [dbc.Col(generate_client_card(client), width="auto") for client in clients],
        justify="start",
        className="g-4",
    ),
])