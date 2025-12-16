# collection of custom plotting functions
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

pio.templates["my_theme"] = go.layout.Template(
    layout=dict(
        font=dict(
            family="Myriad Pro",
            color="black",
        ),
        xaxis=dict(
            showline=True,
            mirror=True,
            linewidth=1.2,
            linecolor="black",
        ),
        yaxis=dict(
            mirror=True,
            showline=True,
            linewidth=1.2,
            linecolor="black",
        ),
    ),
)

pio.templates.default = "plotly_white+my_theme"
pio.renderers.default = "notebook"

colors = ["#777777", "#af9da6", "#c4b4a1", "#214e7b", "#2f6c50"]


def distplot(df, rowcol, title, xtitles, xanot=1.10, vspace=None):
    stat_types = df[rowcol].unique()
    n_stats = len(stat_types)
    if vspace is None:
        vspace = 0.6 / n_stats

    fig = make_subplots(
        rows=n_stats,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=vspace,
    )

    for idx, stat in enumerate(stat_types, start=1):
        df_stat = df[df[rowcol] == stat]
        box_fig = px.box(
            df_stat,
            x="value",
            y="data",
            color="data",
            orientation="h",
            color_discrete_sequence=colors,
        ).update_traces(boxmean=True, line_width=1, marker_size=4)
        for trace in box_fig.data:
            fig.add_trace(trace, row=idx, col=1)

        if df_stat["value"].min() >= 0:
            xmin = 0
            xmax = df_stat["value"].max()
        else:
            xmax = df_stat["value"].abs().max()
            xmin = xmax * -1

        # add mean of each data type on the right side
        means = df_stat.groupby("data", observed=True)["value"].mean()
        for data_name, mean_val in means.items():
            fig.add_annotation(
                xref="paper",
                yref=f"y{idx}",
                x=xanot,
                y=data_name,
                text=f"{mean_val:.3f}",
                showarrow=False,
                font=dict(size=14),
                xanchor="right",
                yanchor="middle",
                align="right",
            )

        # add mean line for Random data
        xvline = df_stat[df_stat["data"] == "Random"]["value"].mean()
        fig.add_vline(
            x=xvline,
            line_dash="dot",
            line_color="red",
            row=idx,
            col=1,
        )
        fig.update_xaxes(
            range=[xmin, xmax * 1.01],
            title=dict(text=xtitles[idx - 1], font_size=18),
            row=idx,
            col=1,
        )
        fig.update_yaxes(title_text=stat)

    fig.update_layout(
        width=900,
        height=165 * n_stats,
        showlegend=False,
        margin=dict(l=50, r=80, t=50, b=70),
    )
    fig.add_annotation(
        text="Mean",
        xref="paper",
        yref="paper",
        x=xanot,
        y=1.01,
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        font=dict(size=18),
    )

    fig.update_xaxes(matches=None)

    def color(text, i):
        if i > 2:
            color = colors[i]
            s = f"<b><span style='color:{str(color)}'> {str(text)} </span></b>"
            return s
        return text

    ticktext = [color(text, i) for i, text in enumerate(df["data"].unique())]
    fig.update_yaxes(
        title=None,
        autorange="reversed",
        tickmode="array",
        tickvals=list(range(len(ticktext))),
        ticktext=ticktext,
        tickfont=dict(size=14),
    )
    fig.update_traces(marker=dict(symbol="circle-open"))
    return fig
