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
pn.extension(sizing_mode="stretch_width")


acc_color = "#001b30"
minor_color = "#b0b2b8"
highlight_color = "#254928"
color3 = "#76a8d9"


expected_model_version = "1.0"
# model_path = "./models/ski_resort_pricing_model.pkl"
# ski_data = pd.read_csv("./data/ski_data_step3_features.csv")


def get_model(web_url):
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


# get model and data
model = get_model(
    "https://storage.googleapis.com/big_mountain_resort/ski_resort_pricing_model.pkl"
)
ski_data = pd.read_csv(
    "https://storage.googleapis.com/big_mountain_resort/ski_data_step3_features.csv"
)


# Drop the extra feature added from last notebook
ski_data = ski_data.drop(columns="log_total_chairs_runs_prod")
# isolate the target variable
big_mountain = ski_data[ski_data.Name == "Big Mountain Resort"]

X = ski_data.loc[ski_data.Name != "Big Mountain Resort", model.X_columns]
y = ski_data.loc[ski_data.Name != "Big Mountain Resort", "AdultWeekend"]

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
        return f"{num:.1f}"


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
    name="Δ Vertical Drop", start=0, end=300, step=150, value=150, **widget_props
)
delta_runs = pnw.IntSlider(
    name="Δ Number of Runs", start=-10, end=1, step=5, value=-10, **widget_props
)
delta_chairs = pnw.IntSlider(
    name="Δ Total Chairs", start=0, end=2, step=1, value=1, **widget_props
)
delta_fastQuads = pnw.IntSlider(
    name="Δ Fast Quads", start=0, end=2, step=1, value=0, **widget_props
)
delta_SnowMaking_ac = pnw.IntSlider(
    name="Δ Snow covered acreage", start=0, end=4, step=2, value=2, **widget_props
)
delta_longest_run = pnw.FloatSlider(
    name="Δ Longest Run", start=0, end=0.4, step=0.2, value=0.2, **widget_props
)

# Estimates for number of guests and days open
n_guests = pnw.IntSlider(
    name="Number of guests",
    start=300_000,
    end=400_000,
    step=50_000,
    value=350_000,
    **widget_props,
)
n_days = pnw.IntSlider(
    name="Number of days", start=1, end=5, step=1, value=5, **widget_props
)


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


# create separate function for last year comparison
def last_year_comparison(n_guests, n_days, ticket_change):
    """Returns a panel Markdown element of the predicted price and last year comparison"""
    new_price = base_price + ticket_change
    new_revenue = new_price * n_guests * n_days
    revenue_change = new_revenue - last_year_revenue

    ticket_color = "darkred" if ticket_change < 0 else minor_color
    revenue_color = "darkred" if revenue_change < 0 else minor_color

    return pn.Row(
        pn.pane.Markdown(
            f"""
        ## Model's relative price shift:
        ### Predicted Δ ticket price:
        # <span style="color:{ticket_color}">${ticket_change:.2f}<span>
        ### Predicted Δ ticket revenue: 
        # <span style="color:{revenue_color}">${format_M(revenue_change)}<span>
        ### Predicted ticket revenue: 
        # <span style="color:{revenue_color}">${format_M(new_revenue)}<span>
        """
        ),
        pn.pane.Markdown(
            f"""
        ## Calculations based on:
        #### Assumed new ticket price:
        ### <span style="color:{ticket_color}">${new_price:.2f}<span>
        #### Estimated number of guests:
        ### <span style="color:{ticket_color}">{n_guests:,.0f}<span> guests
        #### Estimated avg. length of stay: 
        ### <span style="color:{ticket_color}">{n_days:,.0f} <span> days 
        ## Last year's for reference:
        ### Last year's ticket price:
        ### <span style="color:{ticket_color}">${base_price:.2f}<span>
        ### Last year's ticket revenue:
        ### <span style="color:{ticket_color}">${format_M(last_year_revenue)}<span>
        """
        ),
    )


# get df of last year comparison
def get_revenue_df(n_guests, n_days, ticket_change):
    """Get a dataframe of last year's revenue and predicted revenue"""
    new_price = base_price + ticket_change
    predicted_revenue = new_price * n_guests * n_days
    # revenue_change = predicted_revenue - last_year_revenue

    revenue_df = pd.DataFrame(
        {
            "Last Year": [last_year_revenue],
            "Predicted": [predicted_revenue],
        },
        index=["revenue"]
        # orient="columns", columns=["revenue_last_year", "predicted_revenue"]
    )
    return revenue_df


# define function to plot barh on a panel
def barh_callback(df: pd.DataFrame):
    """Returns a hvplot barh element of the revenue comparison dataframe"""
    # create a color column with the color minor color if the value is negative
    dataframe = pd.DataFrame()

    dataframe = df.copy()

    bar_color = (
        "darkred"
        if (
            dataframe["predicted_revenue"].iloc[0]
            < dataframe["revenue_last_year"].iloc[0]
        )
        else minor_color
    )

    bars = hv.Bars(dataframe)
    bars[["revenue_last_year", "predicted_revenue"]]
    bars.redim.label(revenue_last_year="Last Year", predicted_revenue="Predicted")
    return bars[["revenue_last_year", "predicted_revenue"]].opts(
        # return dataframe.hvplot.barh(
        title="Revenue Comparison",
        # xlabel="",
        ylabel="",
        color=bar_color,
        # height=200,
        tools=["hover"],
        yticks=None,
        # yaxis="bare",
        yformatter=NumeralTickFormatter(format="$ 0.0 a"),
        toolbar=None,
    )


def hbar_callback(df):
    # create a color column with the color minor color if the value is negative
    dataframe = pd.DataFrame()
    dataframe = df.copy()
    bar_color = (
        "darkred"
        if (dataframe["Predicted"].iloc[0] < dataframe["Last Year"].iloc[0])
        else color3
    )

    dataframe = dataframe.T.reset_index()
    bars = hv.Bars(dataframe, ["index"], vdims=["revenue"]).opts(
        title="Revenue Comparison",
        color=bar_color,
        xlabel="",
        ylabel="",
        invert_axes=True,
        tools=["hover"],
        toolbar="above",
        xticks=4,
        ylim=(0, 220_000_000),
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


# create a reactive function to output the predicted price and lasr year comparison
reactive_predicted_increase = pn.bind(
    predicted_increase,
    vertical_drop,
    delta_runs,
    delta_chairs,
    delta_fastQuads,
    delta_SnowMaking_ac,
    delta_longest_run,
)

reactive_last_year_comparison = pn.bind(
    last_year_comparison, n_guests, n_days, reactive_predicted_increase
)
reactive_revenue_df = pn.bind(
    get_revenue_df, n_guests, n_days, reactive_predicted_increase
)
reactive_barh = pn.bind(barh_callback, reactive_revenue_df)

reactive_hbar = pn.bind(hbar_callback, reactive_revenue_df)


reactive_panel = pn.Column(
    # reactive_predicted_increase,
    reactive_last_year_comparison,
    # reactive_barh,
    reactive_hbar,
)


# Kill any running servers before starting the new one
pn.state.kill_all_servers()

logo_path = "./images/Big Montain Resort.svg"


sidebar_widgets = [
    pn.WidgetBox(
        "## Feature Adjustments",
        vertical_drop,
        delta_runs,
        delta_chairs,
        delta_fastQuads,
        delta_SnowMaking_ac,
        delta_longest_run,
    ),
    pn.WidgetBox(
        "## Guests Estimates",
        n_guests,
        n_days,
    ),
]


bmr_app = pn.template.FastListTemplate(
    title=f"BMR: Δ TICKET PRICE MODEL",
    sidebar=sidebar_widgets,
    header_color=acc_color,
    accent_base_color=acc_color,
    neutral_color=acc_color,
    logo=logo_path,
)

bmr_app.main.append(reactive_panel)

bmr_app.servable()
