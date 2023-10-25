# import os
import pickle
from urllib.request import urlopen
from bokeh.models import NumeralTickFormatter
import hvplot.pandas  # noqa
import holoviews as hv

import pandas as pd
from panel import widgets as pnw

from sklearn import __version__ as sklearn_version
from sklearn.model_selection import cross_validate
import panel as pn

# throttle the panel widgets to prevent too many events
pn.config.throttled = True

#  set the sizing mode for the panels
pn.extension("floatpanel", sizing_mode="stretch_width")


acc_color = "#001b30"
minor_color = "#b0b2b8"
highlight_color = "#254928"
color3 = "#76a8d9"
neg_color = "#ffa500"


expected_model_version = "1.0"
# model_path = "./models/ski_resort_pricing_model.pkl"
# ski_data = pd.read_csv("./data/ski_data_step3_features.csv")


@pn.cache(per_session=True)
def fetch_model(web_url):
    """Get the model from the web_url and return the model object."""

    response = urlopen(web_url)
    if response.status == 200:
        print("Model found, loading model...")
        model = pickle.load(response)
        if model.version != expected_model_version:
            print("Expected model version doesn't match version loaded")
        if model.sklearn_version != sklearn_version:
            print("Warning: model created under different sklearn version")

        return model
    print("Model not found")
    return None


@pn.cache(per_session=True)
def fetch_data(web_url):
    """gets the dataframe from the web_url and returns the dataframe object."""
    return pd.read_csv(web_url)


model_url = (
    "https://storage.googleapis.com/big_mountain_resort/ski_resort_pricing_model.pkl"
)
data_url = (
    "https://storage.googleapis.com/big_mountain_resort/ski_data_step3_features.csv"
)


# get model and data
model = fetch_model(model_url)
ski_data = fetch_data(data_url)


# Drop the extra feature added from notebook
ski_data = ski_data.drop(columns="log_total_chairs_runs_prod")
# isolate the target variable
big_mountain = ski_data[ski_data.Name == "Big Mountain Resort"]

X = ski_data.loc[ski_data.Name != "Big Mountain Resort", model.X_columns]
y = ski_data.loc[ski_data.Name != "Big Mountain Resort", "AdultWeekend"]
# fit model
model.fit(X, y)

cv_results = cross_validate(
    model, X, y, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1
)
X_bm = ski_data.loc[ski_data.Name == "Big Mountain Resort", model.X_columns]
y_bm = ski_data.loc[ski_data.Name == "Big Mountain Resort", "AdultWeekend"]

bm_pred = model.predict(X_bm).item()

y_bm = y_bm.values.item()


def predict_increase(features, deltas):
    """Increase in modelled ticket price by applying delta to feature.

    Arguments:
    features - list, names of the features in the ski_data dataframe to change
    deltas - list, the amounts by which to increase the values of the features

    Outputs:
    Amount of increase in the predicted ticket price
    """

    bm2 = X_bm.copy()
    for f, d in zip(features, deltas):
        bm2[f] += d
    return model.predict(bm2).item() - model.predict(X_bm).item()


def format_M(num):
    if abs(num) >= 1_000_000:
        return f"{num/1000000:.1f}M"
    else:
        return f"{num:,.1f}"


def hook(plot, element):
    plot.handles["xaxis"].major_tick_line_color = None
    plot.handles["yaxis"].major_tick_line_color = None
    plot.handles["xaxis"].minor_tick_line_color = None
    plot.handles["yaxis"].minor_tick_line_color = None
    plot.handles["xaxis"].axis_line_color = None
    plot.handles["yaxis"].axis_line_color = None


# Some default values for the widgets
base_price, last_year_n_guests, last_year_n_days = 81, 350_000, 5
last_year_revenue = base_price * last_year_n_guests * last_year_n_days

widget_height = 60
widget_props = dict(height=60, bar_color=acc_color)
# declare widgets for feature changes
vertical_drop = pnw.IntSlider(
    name="Δ Vertical Drop", start=0, end=300, step=50, value=0, **widget_props
)
delta_runs = pnw.IntSlider(
    name="Δ Number of Runs", start=-10, end=10, step=1, value=0, **widget_props
)
delta_chairs = pnw.IntSlider(
    name="Δ Total Chairs", start=0, end=5, step=1, value=0, **widget_props
)
delta_fastQuads = pnw.IntSlider(
    name="Δ Fast Quads", start=0, end=2, step=1, value=0, **widget_props
)
delta_SnowMaking_ac = pnw.IntSlider(
    name="Δ Snow covered acreage", start=0, end=10, step=1, value=0, **widget_props
)
delta_longest_run = pnw.FloatSlider(
    name="Δ Longest Run", start=0.0, end=1.0, step=0.2, value=0.0, **widget_props
)

# Estimates for number of guests and days open
n_guests = pnw.IntSlider(
    name="Number of guests",
    start=250_000,
    end=500_000,
    step=10_000,
    value=350_000,
    **widget_props,
)
n_days = pnw.IntSlider(
    name="Number of days", start=1, end=10, step=1, value=5, **widget_props
)

# dict to map the feature names to the widget display names
feature_dict = {
    "vertical_drop": "Δ Vertical Drop",
    "Runs": "Δ Number of Runs",
    "total_chairs": "Δ Total Chairs",
    "fastQuads": "Δ Fast Quads",
    "Snow Making_ac": "Δ Snow covered acreage",
    "LongestRun_mi": "Δ Longest Run",
}


# define a function to output the impact of 1 feature change
def feature_impact(feature, delta):
    """Returns the impact of a feature change on the ticket price"""
    return predict_increase([feature], [delta])


# define a function to output the predicted price
def predicted_increase(
    vertical_drop,
    delta_runs,
    delta_chairs,
    delta_fastQuads,
    delta_SnowMaking_ac,
    delta_longest_run,
):
    """Returns the predicted increase in ticket price based on the feature changes"""
    features = [
        "vertical_drop",
        "Runs",
        "total_chairs",
        "fastQuads",
        "Snow Making_ac",
        "LongestRun_mi",
    ]
    deltas = [
        vertical_drop,
        delta_runs,
        delta_chairs,
        delta_fastQuads,
        delta_SnowMaking_ac,
        delta_longest_run,
    ]
    ticket_change = predict_increase(features, deltas)
    return ticket_change


# create a function to determine the color of the values based on the value of the ticket change
def ticket_color(ticket_change):
    """Returns the color of the ticket change value"""
    return neg_color if ticket_change < 0 else color3


# create a func to display the chosen features values from the widgets
# put the values in the color from the ticket_color function
def display_features2(
    vertical_drop,
    delta_runs,
    delta_chairs,
    delta_fastQuads,
    delta_SnowMaking_ac,
    delta_longest_run,
    ticket_change,
):
    """Returns a panel Markdown element of the chosen features values"""
    return pn.pane.Markdown(
        f"""
        |Vertical Drop|Number of Runs|Total Chairs|Fast Quads|Snow covered acreage|Longest Run|
        |---|---|---|---|---|---|
        <span style='color: {ticket_color(ticket_change)}'> {vertical_drop} ft|{delta_runs} runs|{delta_chairs} chairs|{delta_fastQuads} fast quads|{delta_SnowMaking_ac} acres|{delta_longest_run} miles|
        """
    )


def display_features(widgets, ticket_change):
    """Returns a panel Markdown of the widgets values"""
    value_color = ticket_color(ticket_change)
    features = [(w.name, w.value) for w in widgets]
    rows = "\n".join(
        [
            f"|{f[0]}|<span style='color: {value_color}'>{f[1]:,}</span>|"
            for f in features
        ]
    )
    return pn.pane.Markdown(f"{'|'.join(['Feature', 'Value'])}\n|:--|---:|\n{rows}\n")


def display_features_impacts(widgets, ticket_change):
    """Returns a panel Markdown element of the chosen features values"""
    value_color = ticket_color(ticket_change)
    # get the widget_name, widget_value, feature_name from the widgets
    features = [
        (w.name, str(w.value), next(k for k, v in feature_dict.items() if v == w.name))
        for w in widgets
    ]
    rows = "\n".join(
        [
            f"|{f[0]}|<span style='color: {value_color}'>{f[1]}</span>|${feature_impact(f[2], float(f[1])):.2f}|"
            for f in features
        ]
    )
    return pn.pane.Markdown(
        f"{'|'.join(['Feature', 'Value', 'Approx. Impact'])}\n|:--|---:|---:|\n{rows}\n"
    )


# create separate function for last year comparison
def last_year_comparison(n_guests, n_days, ticket_change):
    """Returns a panel Markdown element of the predicted price and last year comparison"""
    new_price = base_price + ticket_change
    new_revenue = new_price * n_guests * n_days
    revenue_change = new_revenue - last_year_revenue

    ticket_color = neg_color if ticket_change < 0 else color3
    revenue_color = neg_color if revenue_change < 0 else color3
    card_opts = dict(hide_header=True, width=300, margin=(0, 1, 0, 10))

    d_ticket_card = pn.Card(
        pn.indicators.Number(
            name="Δ Ticket price",
            value=ticket_change,
            format="${value:.2f}",
            default_color=ticket_color,
        ),
        **card_opts,
    )

    new_ticket_card = pn.Card(
        pn.indicators.Number(
            name="New ticket price",
            value=new_price,
            format="${value:.2f}",
            default_color=ticket_color,
        ),
        **card_opts,
    )

    d_ticket_revenue_card = pn.Card(
        pn.indicators.Number(
            name="Estimated Δ revenue",
            value=revenue_change / 1_000_000
            if abs(revenue_change) >= 1_000_000
            else revenue_change,
            format="${value:.1f}M"
            if abs(revenue_change) >= 1_000_000
            else "${value:,.0f}",
            default_color=revenue_color,
        ),
        **card_opts,
    )

    ticket_revenue_card = pn.Card(
        pn.indicators.Number(
            name="Estimated revenue",
            value=new_revenue / 1_000_000
            if abs(new_revenue) >= 1_000_000
            else new_revenue,
            format="${value:.1f}M"
            if abs(new_revenue) >= 1_000_000
            else "${value:,.0f}",
            default_color=revenue_color,
        ),
        **card_opts,
    )

    last_year_ticket_card = pn.Card(
        pn.indicators.Number(
            name="Last Year ticket price",
            value=base_price,
            format="${value:.2f}",
            default_color=minor_color,
        ),
        **card_opts,
    )

    last_year_revenue_card = pn.Card(
        pn.indicators.Number(
            name="Last Year revenue",
            value=last_year_revenue / 1_000_000
            if abs(last_year_revenue) >= 1_000_000
            else last_year_revenue,
            format="${value:.1f}M"
            if abs(last_year_revenue) >= 1_000_000
            else "${value:,.0f}",
            default_color=minor_color,
        ),
        **card_opts,
    )

    cards = pn.FlexBox(
        *[
            pn.pane.Markdown("### Relative price change:"),
            d_ticket_card,
            new_ticket_card,
            last_year_ticket_card,
            pn.pane.Markdown("### Revenue Estimates:"),
            d_ticket_revenue_card,
            ticket_revenue_card,
            last_year_revenue_card,
        ],
        # align_items="center",
        # align_content='normal',
        justify_content="space-between",
        flex_wrap="wrap",
    )

    return pn.Column(cards)


# get df of last year comparison
def get_revenue_df(n_guests, n_days, ticket_change):
    """Get a dataframe of last year's revenue and predicted revenue"""
    new_price = base_price + ticket_change
    predicted_revenue = new_price * n_guests * n_days
    # revenue_change = predicted_revenue - last_year_revenue

    revenue_df = pd.DataFrame(
        {
            "Last Year": [last_year_revenue],
            "Estimated": [predicted_revenue],
        },
        index=["revenue"]
        # orient="columns", columns=["revenue_last_year", "predicted_revenue"]
    )
    return revenue_df


def hbar_callback(df):
    # create a color column with the color minor color if the value is negative
    dataframe = pd.DataFrame()
    dataframe = df.copy()
    bar_color = (
        neg_color
        if (dataframe["Estimated"].iloc[0] < dataframe["Last Year"].iloc[0])
        else color3
    )

    dataframe = dataframe.T.reset_index()
    dataframe.loc[:, "color"] = minor_color
    dataframe.loc[1, "color"] = bar_color
    bars = hv.Bars(dataframe, ["index"], vdims=["revenue", "color"]).opts(
        title="Revenue Comparison",
        color="color",
        xlabel="",
        ylabel="",
        invert_axes=True,
        tools=["hover"],
        toolbar="above",
        xticks=4,
        # ylim=(0, 220_000_000),
        xformatter=NumeralTickFormatter(format="$ 0.0 a"),
        hooks=[hook],
        fontsize={
            "title": "16pt",
            "labels": "12pt",
            "xticks": "10pt",
            "yticks": "10pt",
        },
    )
    return bars


# define the reset button and callback function
def reset_features(event):
    for widget in feature_widgets:
        widget.value = default_values[widget.name]


tooltip_text = """
Features
- You can adjust the features of the ski resort, such as 
  - vertical drop, 
  - number of runs, 
  - total chairs, 
  - fast quads, 
  - snow covered acreage, and 
  - longest run. 
- These features are based on the data of the market competitors of Big Mountain Resort.

Model: 
- The website uses a machine learning model called Random Forest to predict the ticket price change based on the features you adjust. 
- The model was trained on the data of 330 ski resorts in North America.

Estimates: 
- The website also estimates the revenue change based on the ticket price change and some assumptions about the number of guests and number of days of stay. 
- These assumptions can also be adjusted by you.

Widgets: 
- Widgets panel is a floating panel on the left side of the screen where you can adjust the Features and Estimates. 
- As you adjust them, you will see the results on the right side of the screen.

"""
info_window = pn.pane.Markdown(tooltip_text)

reset_button = pn.widgets.Button(name="Reset")
reset_button.on_click(reset_features)

info_icon = pnw.TooltipIcon(value=tooltip_text)
info_bar = pn.Row(
    pn.Spacer(width=300),info_icon,
)


# create a reactive function to output the predicted price and last year comparison
reactive_predicted_increase = pn.bind(
    predicted_increase,
    vertical_drop,
    delta_runs,
    delta_chairs,
    delta_fastQuads,
    delta_SnowMaking_ac,
    delta_longest_run,
)

reactive_color = pn.bind(ticket_color, reactive_predicted_increase)

feature_widgets = [
    vertical_drop,
    delta_runs,
    delta_chairs,
    delta_fastQuads,
    delta_SnowMaking_ac,
    delta_longest_run,
]

default_values = {w.name: w.value for w in feature_widgets}
estimator_widgets = [n_guests, n_days]

# reactive features display
reactive_features = pn.bind(
    display_features_impacts, feature_widgets, reactive_predicted_increase
)
reactive_estimators = pn.bind(
    display_features, estimator_widgets, reactive_predicted_increase
)


# bound the reactive function to the last year comparison function
reactive_last_year_comparison = pn.bind(
    last_year_comparison, n_guests, n_days, reactive_predicted_increase
)
reactive_revenue_df = pn.bind(
    get_revenue_df, n_guests, n_days, reactive_predicted_increase
)

reactive_hbar = pn.bind(hbar_callback, df=reactive_revenue_df)
features_table_md = pn.Column(
    pn.panel(pn.pane.Markdown("#### Whatif Features"), width=300),
    reactive_features,
    width=300,
)
estimates_table_md = pn.Column(
    pn.panel(pn.pane.Markdown("#### Estimates"), width=300),
    reactive_estimators,
    width=300,
)


# Kill any running servers before starting the new one
pn.state.kill_all_servers()

logo_path = "./images/Big Montain Resort.svg"

md_tables = [features_table_md, estimates_table_md]

w_controls = [
    reset_button,
    pn.Card(
        vertical_drop,
        delta_runs,
        delta_chairs,
        delta_fastQuads,
        delta_SnowMaking_ac,
        delta_longest_run,
        title="Features",
    ),
    pn.Card(n_guests, n_days, title="Estimates"),
]

w_floatie = pn.layout.FloatPanel(
    *w_controls,
    name="Widgets",
    contained=False,
    position="left-center",
    width=300,
    config={
        "headerControls": {"close": "remove", "maximize": "remove"},
        "borderRadius": ".5rem",
        },
    theme=color3,
)

intro = pn.pane.HTML(
    """<p align='right'><i>Skiing is a slippery slope: let's slide!</i></p>
    This is <b>SlideRuleBMR</b>!
        
    <p>Curious if adding that extra &#xBD; mile to your longest run will offset
    closing 10 other runs?</p>
    
    Let's find out!<br>
    
    <p>Use the widgets in the floating pane to the left of your screen to get
    started!</p>
    
    <p>But first, close this window by clicking the <b>'X'</b> in the top right corner.</p>
    """
    # Are you curious as to what an appropriate price increase for an exciting addition
    # to **Big Mountain Resort** that you know customers will say yes for?
    # Or, are you wondering about what adds little-to-no value to the ticket price,
    # that which you can wipeout! Maybe both?
    # If you said 'yes', 'no', or 'maybe', to any of those questions proceed and enjoy!
    # At your fingertips, you are able to create some of your most curious what-if
    # scenarios and see what impact they have on the ticket price using
    # Machine Learning and the **Random Forest** algorithm built and trained specifically for
    # Big Mountain Resort on its market competitors data.
)

intro_floatie = pn.layout.FloatPanel(
    pn.panel(intro),
    name=f"Welcome!",
    # theme=color3,
    contained=False,
    position="center",
    width=500,
    config={
        "headerControls": {"maximize": "remove"},
        "theme": {
            "bgPanel": color3,
            "border": f"thin solid {minor_color}",
            "colorHeader": acc_color,
        },
        "colorContent": "f00",
        "borderRadius": ".5rem",
    },
)


reactive_panel = pn.Column(info_icon, reactive_last_year_comparison, reactive_hbar)

bmr_app = pn.template.FastListTemplate(
    title=f"SlideRuleBMR: WHATIF ESTIMATOR",
    header_color=acc_color,
    header_background=color3,
    accent_base_color=acc_color,
    neutral_color=acc_color,
    logo=logo_path,
)


bmr_app.sidebar.extend(md_tables)

bmr_app.main.append(reactive_panel)
bmr_app.sidebar.append(pn.Column(w_floatie))
bmr_app.main.append(pn.Column(intro_floatie, sizing_mode="stretch_width"))

bmr_app.servable()
