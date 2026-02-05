import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG], use_pages=True)
navbar = dbc.NavbarSimple(
    children=[
        *[
        dbc.NavItem(dbc.NavLink(f"{page['name']}", href=page["relative_path"]))
         for page in dash.page_registry.values() if not page["relative_path"] in ["/client"]
        ],
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("More pages", header=True),
                dbc.DropdownMenuItem("Page 2", href="#"),
                dbc.DropdownMenuItem("Page 3", href="#"),
            ],
            nav=True,
            in_navbar=True,
            label="More",
        ),
    ],
    brand="Denise Khuu",
    brand_href="#",
    color="dark",
    dark=True,
)

app.layout = html.Div([
    navbar,
    dbc.Container(dash.page_container)
])



if __name__ == '__main__':
    app.run(debug=True)