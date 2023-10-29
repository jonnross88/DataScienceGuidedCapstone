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
    elif abs(num) >= 1_000:
        return f"{num/1000:.0f}K"
    else:
        return f"{num:,.1f}"


def hook(plot, element):
    # plot.handles["xaxis"].major_tick_line_color = None
    plot.handles["yaxis"].major_tick_line_color = None
    plot.handles["xaxis"].minor_tick_line_color = None
    plot.handles["yaxis"].minor_tick_line_color = None
    plot.handles["xaxis"].axis_line_color = None
    plot.handles["yaxis"].axis_line_color = None


base_price, last_year_n_guests, last_year_n_days = 81, 350_000, 5
last_year_revenue = base_price * last_year_n_guests * last_year_n_days

# Some default values for the widgets
widget_height = 60
widget_props = dict(height=60, bar_color=acc_color)

# declare widgets for feature changes
vertical_drop = pnw.IntSlider(
    name="Δ Vertical Drop", start=-300, end=300, step=50, value=0, **widget_props
)
delta_runs = pnw.IntSlider(
    name="Δ Number of Runs", start=-10, end=10, step=1, value=0, **widget_props
)
delta_chairs = pnw.IntSlider(
    name="Δ Total Chairs", start=-5, end=5, step=1, value=0, **widget_props
)
delta_fastQuads = pnw.IntSlider(
    name="Δ Fast Quads", start=-2, end=2, step=1, value=0, **widget_props
)
delta_SnowMaking_ac = pnw.IntSlider(
    name="Δ Snow covered acreage", start=-5, end=5, step=1, value=0, **widget_props
)
delta_longest_run = pnw.FloatSlider(
    name="Δ Longest Run", start=-1.0, end=1.0, step=0.1, value=0.0, **widget_props
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
addl_cost_toggle = pnw.Checkbox(name="Include Add'l Cost", value=False)
addl_cost = pnw.IntSlider(
    name="Add'l Cost", value=0, start=0, end=5_000_000, step=10_000, visible=False
)
addl_cost_text = pnw.StaticText(name="Additional Cost", value=f"${addl_cost.value:,}")

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
# def ticket_color(ticket_change):
#     """Returns the color of the ticket change value"""
#     return neg_color if ticket_change < 0 else color3
# define function to output the revenue estimates
def income_estimates(guests, days, ticket_change, expenses):
    new_price = base_price + ticket_change
    return (new_price * guests * days) - expenses


def display_features(widgets, new_income):
    """Returns a panel Markdown of the widgets values"""
    income_change = new_income - last_year_revenue
    value_color = neg_color if income_change < 0 else color3
    features = [(w.name, w.value) for w in widgets]
    rows = "\n".join(
        [
            f"|{f[0]}|<span style='color: {value_color}'>{f[1]:,}</span>|"
            for f in features
        ]
    )
    return pn.pane.Markdown(
        f"{'|'.join(['Contributor', 'Value'])}\n|:--|---:|\n{rows}\n"
    )


def display_features_impacts(widgets, ticket_change):
    """Returns a panel Markdown element of the chosen features values"""
    value_color = neg_color if ticket_change < 0 else color3
    # get the widget_name, widget_value, feature_name from the widgets
    features = [
        (
            w.name,
            f"{w.value:.1f}" if isinstance(w.value, float) else w.value,
            next(k for k, v in feature_dict.items() if v == w.name),
            feature_impact(
                next(k for k, v in feature_dict.items() if v == w.name), float(w.value)
            ),
        )
        for w in widgets
    ]
    total_impact = sum(abs(f[3]) for f in features)
    rows = "\n".join(
        [
            f"|{f[0]}|<span style='color: {value_color}'>{f[1]}</span>|${f[3]:.2f}|{abs(f[3])/(0.000001+total_impact):.0%}|"
            for f in features
        ]
    )
    return pn.pane.Markdown(
        f"{'|'.join(['Feature', 'Change', 'Isolated Impact', '% of Total Impact'])}\n|:--|---:|---:|---:|\n{rows}\n"
    )


last_addl_cost = {"value": addl_cost.value}


# use the expense toggle to show or hide the expense input
def expense_toggle_callback(event):
    if addl_cost_toggle.value:
        addl_cost.visible = True
        addl_cost.value = last_addl_cost["value"]
    else:
        addl_cost.visible = False
        last_addl_cost["value"] = addl_cost.value
        addl_cost.value = 0


addl_cost_toggle.param.watch(expense_toggle_callback, "value")


def static_text_callback(event):
    addl_cost_text.value = f"${addl_cost.value:,}"


addl_cost.param.watch(static_text_callback, "value")


# create separate function for last year comparison
def last_year_comparison(guests, days, ticket_change, expenses=0):
    """Returns a panel Markdown element of the predicted price and last year comparison"""
    new_price = base_price + ticket_change
    new_revenue = new_price * guests * days
    new_income = new_revenue - expenses
    income_change = new_income - last_year_revenue

    price_color = neg_color if ticket_change < 0 else color3
    income_color = neg_color if income_change < 0 else color3
    card_opts = dict(hide_header=True, width=300)
    num_opts = dict(font_size="40pt")

    sub_heads = []
    sub_heads.append(pn.pane.Markdown("### Ticket price:"))
    sub_heads.append(pn.pane.Markdown("### Aggregates:"))

    indicators = []
    delta_ticket = pn.indicators.Number(
        name="Δ Ticket price",
        value=ticket_change,
        format="${value:.2f}",
        default_color=price_color,
        **num_opts,
    )
    indicators.append(delta_ticket)
    new_ticket = pn.indicators.Number(
        name="New ticket price",
        value=new_price,
        format="${value:.2f}",
        default_color=price_color,
        **num_opts,
    )
    indicators.append(new_ticket)
    last_year_ticket = pn.indicators.Number(
        name="Last Year ticket price",
        value=base_price,
        format="${value:.2f}",
        default_color=minor_color,
        **num_opts,
    )
    indicators.append(last_year_ticket)

    d_revenue = pn.indicators.Number(
        name="Estimated Δ Income",
        value=income_change / 1_000_000
        if abs(income_change) >= 1_000_000
        else income_change,
        format="${value:.1f}M" if abs(income_change) >= 1_000_000 else "${value:,.0f}",
        default_color=income_color,
        **num_opts,
    )
    indicators.append(d_revenue)
    revenue = pn.indicators.Number(
        name="Estimated Revenue",
        value=new_revenue / 1_000_000 if abs(new_revenue) >= 1_000_000 else new_revenue,
        format="${value:.1f}M" if abs(new_revenue) >= 1_000_000 else "${value:,.0f}",
        default_color=income_color,
        **num_opts,
    )
    indicators.append(revenue)

    last_year_revenue_i = pn.indicators.Number(
        name="Last Year Revenue",
        value=last_year_revenue / 1_000_000
        if abs(last_year_revenue) >= 1_000_000
        else last_year_revenue,
        format="${value:.1f}M"
        if abs(last_year_revenue) >= 1_000_000
        else "${value:,.0f}",
        default_color=minor_color,
        **num_opts,
    )
    indicators.append(last_year_revenue_i)

    # put each indicator on a card
    indicator_cards = [pn.Card(indicator, **card_opts) for indicator in indicators]
    # add i subhead for every 3 indicators
    mixed_list = (
        [sub_heads[0]] + indicator_cards[:3] + [sub_heads[1]] + indicator_cards[3:]
    )

    cards = pn.FlexBox(*mixed_list, justify_content="space-between", flex_wrap="wrap")

    return pn.Column(cards)


# get df of last year comparison
def get_revenue_df(guests, days, ticket_change):
    """Get a dataframe of last year's revenue and predicted revenue"""
    new_price = base_price + ticket_change
    predicted_revenue = new_price * guests * days
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
        active_tools=["box_zoom"],
        toolbar="above",
        xticks=4,
        # ylim=(0, 220_000_000),
        xformatter=NumeralTickFormatter(format="$ 0 a"),
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
Widgets: 
- Widgets controls are on a floating panel found initially on the left side of the screen
- Using these controls, you can adjust the Features and Estimates. 

Features
You can adjust the features of the ski resort, such as 
  - fast quads, 
  - number of runs, 
  - snow covered acreage, and 
  - vertical drop, 
  - total chairs, 
  - longest run. 
These features either had the highest feature importance from the modal or are features of interest to Big Mountain Resort.

Model: 
- As you adjust any of the features/ widget values, the model predicts the relative combined change to the ticket prices accordingly
- The individual impact of any of the adjusted features if taken alone are show in the table in the sidebar

Estimates: 
- Estimated changes in revenue based on the ticket price change are also extrapolated
- The revenue changes uses last year's guest turnout by default.
- Controls to edit these values for this year's guest turnout can also be adjusted from the widget panel.


"""
info_window = pn.pane.Markdown(tooltip_text)

reset_button = pn.widgets.Button(name="Reset")
reset_button.on_click(reset_features)

main_info_icon = pnw.TooltipIcon(value=tooltip_text)
main_info_bar = pn.Row(
    pn.Spacer(width=300),
    main_info_icon,
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

reactive_income_estimates = pn.bind(
    income_estimates, n_guests, n_days, reactive_predicted_increase, addl_cost
)

# reactive_color = pn.bind(ticket_color, reactive_predicted_increase)

feature_widgets = [
    delta_fastQuads,
    delta_runs,
    delta_SnowMaking_ac,
    vertical_drop,
    delta_chairs,
    delta_longest_run,
]

default_values = {w.name: w.value for w in feature_widgets}
estimator_widgets = [n_guests, n_days]
addl_cost_widgets = [addl_cost]


# reactive features display
reactive_features = pn.bind(
    display_features_impacts, feature_widgets, reactive_predicted_increase
)
reactive_estimators = pn.bind(
    display_features, estimator_widgets, reactive_income_estimates
)
reactive_addl_cost = pn.bind(
    display_features, addl_cost_widgets, reactive_income_estimates
)
reactive_pie_chart = pn.bind(pie_chart_callback, feature_widgets)


# bound the reactive function to the last year comparison function
reactive_last_year_comparison = pn.bind(
    last_year_comparison, n_guests, n_days, reactive_predicted_increase, addl_cost
)
reactive_revenue_df = pn.bind(
    get_revenue_df, n_guests, n_days, reactive_predicted_increase
)

reactive_hbar = pn.bind(hbar_callback, df=reactive_revenue_df)

# Markdown tables in side panel
features_table_md = pn.Column(
    pn.Row(
        pn.pane.Markdown("#### Whatif Features"),
        pnw.TooltipIcon(
            value="""
            These are the features which have been adjusted from the widgets panel. 
            Note that the sum of the individual impacts may not equal the total 
            impact due to interactions between the features."""
        ),
        width=300,
    ),
    reactive_features,
    width=300,
)
estimates_table_md = pn.Column(
    pn.panel(pn.pane.Markdown("#### Estimates"), width=300),
    reactive_estimators,
    width=300,
)
addl_cost_table_md = pn.Column(
    pn.panel(pn.pane.Markdown("#### Additional Cost"), width=300),
    reactive_addl_cost,
    width=300,
)


# Kill any running servers before starting the new one
pn.state.kill_all_servers()

logo_path = "./Big Mountain Resort.svg"

md_tables = [features_table_md, estimates_table_md, addl_cost_table_md]

# Widgets Controls
w_controls = [
    reset_button,
    pn.Card(
        delta_fastQuads,
        delta_runs,
        delta_SnowMaking_ac,
        vertical_drop,
        delta_chairs,
        delta_longest_run,
        title="Features",
    ),
    pn.Card(n_guests, n_days, title="Estimates"),
    pn.Card(addl_cost_toggle, addl_cost_text, addl_cost, title="Add'l Cost"),
    pn.Card(reactive_pie_chart, title="Feature Impacts")
]
# declare floating widget controls panel
w_floatie = pn.layout.FloatPanel(
    name="Widgets",
    contained=False,
    position="left-center",
    width=300,
    offsetx=20,
    height=300,
    config={
        "headerControls": {"close": "remove", "maximize": "remove"},
        "borderRadius": ".5rem",
    },
    theme=color3,
)
w_floatie.append(pn.Column(*w_controls, sizing_mode="stretch_width"))

intro = pn.pane.HTML(
    f"""<p align='right'><i>Skiing is a slippery slope: let's slide!</i></p>
    <div>This is <span style='color: {color3}'><b>SlideRuleBMR</b></span>!</div>
        
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
intro_floatie.append(pn.Card(intro, hide_header=True, sizing_mode="stretch_width"))

reactive_panel = pn.Column(main_info_icon, reactive_last_year_comparison, reactive_hbar)

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
bmr_app.sidebar.append(pn.Column(intro_floatie, sizing_mode="stretch_width"))

bmr_app.servable()
