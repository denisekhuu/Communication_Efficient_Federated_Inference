from dash import html, dcc, Input, Output, callback
import dash
import dash_bootstrap_components as dbc
from common.main import number_of_clients, transformed_data, op_transform, grouped_sets, clients
from urllib.parse import parse_qs, urlparse
from common.utils import image_to_base64 

dash.register_page(__name__, path="/client")

layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="client-output")
])

def create_overview_table(client):
    row1 = html.Tr([html.Td("Client Id"), html.Td(client.id)])
    row2 = html.Tr([html.Td("Trainset Size"), html.Td(len(client.trainset))])
    row3 = html.Tr([html.Td("Testset Size"), html.Td(len(client.testset))])
    row4 = html.Tr([html.Td("Sensor Coverage (in %)"), html.Td(op_transform.coverage_sensors)])

    table_body = [html.Tbody([row1, row2, row3, row4])]

    return dbc.Table(
        table_body,
        bordered=True,
        color="secondary",
        hover=True,
        responsive=True,
        striped=True,
    )  

def create_tabs_for_dataset(client):
    label_to_indices = grouped_sets

    tabs = []
    for label, indices in label_to_indices.items():
        images = [image_to_base64(client.trainset[indices[0][i]][0]) for i in range(9)]
        image_divs = [
            html.Div([
                html.Img(src=f"data:image/png;base64,{img}", style={"height": "100px"}),
            ], style={"margin": "10px", "textAlign": "center"})
            for i, img in zip(range(len(images)), images)
        ]

        # To change grid of example
        tab_content = html.Div(
            image_divs,
                    style={
            "display": "grid",
            "gridTemplateColumns": "repeat(3, 1fr)",
            "gap": "2px",
            "justifyItems": "center",
            "width": "25%",
        }
        )
        
        tabs.append(dcc.Tab(label=f"Label {label}", children=[tab_content]))

    return dcc.Tabs(tabs)

@callback(
    Output("client-output", "children"),
    Input("url", "search")
)
def display_client_page(search):
    query_params = parse_qs(urlparse(search).query)
    client_id = query_params.get("id", [None])[0]

    if not client_id:
        return "No client ID provided."
    
    client_id = int(client_id) if client_id else 0
    client = clients[client_id]
    if client:
        return html.Div([
            html.H3(f"Overview: Client {int(client_id)}"),
            create_overview_table(client), 
            html.Hr(),
            html.H3("Example"),
            create_tabs_for_dataset(client),
        ])
    else:
        return html.P("Client not found.")