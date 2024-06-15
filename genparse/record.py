from .util import format_table
import pandas as pd
import numpy as np


class SMCRecord(dict):
    """
    For storing and visualizing information about a run of SMC (an ad-hoc trace).

    Methods:
        __init__(self):
            Initializes the SMCRecord instance. Accepts any positional and keyword arguments
            that are valid for the built-in dict class.

        to_df(self, process=True):
            Transforms record to pandas dataframe. If process==True, adds additional columns that may be useful

        plot_particles_trajectory():
            Experimental plotting function for whole record
            Requires `plotly` to be installed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_particles = len(
            self["weight"][0]
        )  # any more transparent way of getting this would smarter

    def __repr__(self):
        return f"SMCRecord({super().__repr__()})"

    def _repr_html_(self):
        # Just copied from chart.Chart  --- Might not look good.
        return (
            '<div>SMCRecord:<div><div style="font-family: Monospace;">'
            + format_table(self.to_df().items(), headings=["key", "value"])
            + "</div>"
        )

    def to_df(self, process=True):
        df = pd.DataFrame(self)
        if not process:
            return df
        df["token"] = df["context"].apply(
            lambda x: [ri[-1] if len(ri) > 0 else [] for ri in x]
        )
        df["change in w"] = df[
            "average weight"
        ].diff()  # incremental diff in avg weight (biased (over)estimator of marginalizing constant)
        return df

    def df_for_plotting(self):
        # Process the data for plotting.
        recs = (
            self.to_df(process=True)
            .explode(["context", "weight", "resampled as", "token"])
            .reset_index(drop=True)
        )
        recs["particle"] = recs.groupby(recs.index // self.n_particles).cumcount()
        recs["context_string"] = recs["context"].apply(lambda x: "".join(x[:-1]))
        recs["exp_average weight"] = recs["average weight"].apply(lambda x: np.exp(x))
        recs["exp_weight"] = recs["weight"].apply(lambda x: np.exp(x))
        recs["prop_exp_weight"] = recs["exp_weight"] / recs["exp_average weight"]
        return recs

    def plotly1(self, xrange=None):

        import plotly.graph_objects as go
        import plotly.express as px

        d_ = self.df_for_plotting()
        if xrange:
            d_ = d_[(d_["step"] > xrange[0]) & (d_["step"] <= xrange[1])]

        # Define color palette
        pallette = px.colors.n_colors(
            "rgb(255, 0, 0)",
            "rgb(0, 125, 255)",
            len(d_["particle"].unique()),
            colortype="rgb",
        )
        color_map = {
            p: pallette[i % len(pallette)]
            for i, p in enumerate(d_["particle"].unique())
        }

        # Initialize the figure
        fig = go.Figure()

        # Add scatter plot points
        for resampled_as in d_["resampled as"].unique():
            resampled_as_data = d_[d_["resampled as"] == resampled_as]
            fig.add_trace(
                go.Scatter(
                    x=resampled_as_data["step"],
                    y=resampled_as_data["particle"],
                    mode="markers+text",
                    marker=dict(
                        size=np.sqrt(resampled_as_data["prop_exp_weight"] * 200),
                        color=color_map[resampled_as],
                        opacity=0.15,
                    ),
                    text=resampled_as_data["token"],
                    hoverinfo="text",
                    hovertext=resampled_as_data.apply(
                        lambda row: (
                            f"Token:    {'`<b>'+row['token']+'</b>`' if row['token'] else ''}<br>"
                            + f"Context:  {row['context_string']}<br>"
                            + f"Step {row['step']}; Avg weight = {row['average weight']:4f}<br>"
                            + f"Particle {row['particle']}; Weight = {row['weight']:4f}<br>"
                            + f"{'        ↳ resampled as particle '+str(row['resampled as']) if row['resample?'] else ''}"
                        ),
                        axis=1,
                    ),
                    showlegend=False,
                )
            )

        # Get resample or no-resample steps, add vline on the resample ones
        resample_steps = d_[d_["resample?"] == True]["step"].unique()
        for step in resample_steps:
            fig.add_vline(x=step, line_width=4, opacity=0.15, line_color="gray")

        # Add lines connecting nodes between steps to represent resampling
        for resampled_as in d_["resampled as"].unique():
            resampled_as_data = d_[d_["resampled as"] == resampled_as]
            for step in resample_steps:
                for _, row in resampled_as_data[
                    resampled_as_data["step"] == step
                ].iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[row["step"], row["step"] + 1],
                            y=[row["resampled as"], row["particle"]],
                            mode="lines",
                            line=dict(color=color_map[resampled_as]),
                            opacity=0.3,
                            name=resampled_as,
                            showlegend=False,
                        )
                    )

            for step in d_["step"].unique():
                for _, row in resampled_as_data[
                    resampled_as_data["step"] == step
                ].iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[row["step"], row["step"] + 1],
                            y=[row["particle"], row["particle"]],
                            mode="lines",
                            line=dict(color="gray"),
                            opacity=0.15,
                            name=resampled_as,
                            showlegend=False,
                        )
                    )

        # Update layout to include range slider

        # fig.update_traces(textposition='middle right')
        fig.update_layout(
            plot_bgcolor="#fff",
            showlegend=False,
            margin=dict(l=0, r=0, t=15, b=0),
            xaxis=dict(
                # rangeslider=dict(
                #     visible=True,
                #     thickness=0.05,
                #     range=[d_['step'].min(), d_['step'].max()]  # Set initial range
                # )
            ),
            yaxis=dict(showticklabels=False, visible=False),
        )

        return fig

    def plotly2(self, xrange=None, height=None, width=None):

        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots

        d_ = self.df_for_plotting()
        fullrange = [d_["step"].min(), 100]
        if xrange:
            d_ = d_[(d_["step"] > xrange[0]) & (d_["step"] <= xrange[1])]

        # Define color palette
        pallette = px.colors.n_colors(
            "rgb(255, 0, 0)",
            "rgb(0, 125, 255)",
            len(d_["particle"].unique()),
            colortype="rgb",
        )
        color_map = {
            p: pallette[i % len(pallette)]
            for i, p in enumerate(d_["particle"].unique())
        }

        # Initialize the figure
        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.8, 0.2],
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("particles", "est. logprob (change in average weight)"),
        )

        # Add scatter plot points
        for resampled_as in d_["resampled as"].unique():
            resampled_as_data = d_[d_["resampled as"] == resampled_as]
            fig.add_trace(
                go.Scatter(
                    x=resampled_as_data["step"],
                    y=resampled_as_data["particle"],
                    mode="markers+text",
                    marker=dict(
                        size=np.sqrt(resampled_as_data["prop_exp_weight"] * 200),
                        color=color_map[resampled_as],
                        opacity=0.15,
                    ),
                    text=resampled_as_data["token"],
                    hoverinfo="text",
                    hovertext=resampled_as_data.apply(
                        lambda row: (
                            f"Token:    {'`<b>'+row['token']+'</b>`' if row['token'] else ''}<br>"
                            + f"Context:  {row['context_string']}<br>"
                            + f"Step {row['step']}; Avg weight = {row['average weight']:4f}<br>"
                            + f"Particle {row['particle']}; Weight = {row['weight']:4f}<br>"
                            + f"{'        ↳ resampled as particle '+str(row['resampled as']) if row['resample?'] else ''}"
                        ),
                        axis=1,
                    ),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Get resample or no-resample steps, add vline on the resample ones
        resample_steps = d_[d_["resample?"] == True]["step"].unique()
        for step in resample_steps:
            fig.add_vline(x=step, line_width=4, opacity=0.15, line_color="gray")

        # Add lines connecting nodes between steps to represent resampling
        for resampled_as in d_["resampled as"].unique():
            resampled_as_data = d_[d_["resampled as"] == resampled_as]
            for step in resample_steps:
                for _, row in resampled_as_data[
                    resampled_as_data["step"] == step
                ].iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[row["step"], row["step"] + 1],
                            y=[row["resampled as"], row["particle"]],
                            mode="lines",
                            line=dict(color=color_map[resampled_as]),
                            opacity=0.3,
                            name=resampled_as,
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

            for step in d_["step"].unique():
                for _, row in resampled_as_data[
                    resampled_as_data["step"] == step
                ].iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[row["step"], row["step"] + 1],
                            y=[row["particle"], row["particle"]],
                            mode="lines",
                            line=dict(color="gray"),
                            opacity=0.15,
                            name=resampled_as,
                            showlegend=False,
                        ),
                        row=1,
                        col=1,
                    )

        fig.append_trace(
            go.Scatter(
                x=d_["step"],
                y=d_["change in w"],
                line=dict(color="green"),
                mode="lines+markers",
                hoverinfo="text",
                hovertext=d_.apply(lambda row: row["change in w"], axis=1),
            ),
            row=2,
            col=1,
        )

        # Update layout to include range slider

        # fig.update_traces(textposition='middle right')
        fig.update_layout(
            plot_bgcolor="#fff",
            showlegend=False,
            margin=dict(l=40, r=40, t=40, b=40),
            height=height,
            width=width,
            xaxis_range=fullrange,
            # xaxis=dict(
            #     rangeslider=dict(
            #         visible=True,
            #         thickness=0.02,
            #         # range=[d_['step'].min(), d_['step'].max()]  # Set initial range
            #     )
            # ),
            # yaxis=dict(
            #     showticklabels=False,
            #     visible=False
            # )
        )

        return fig

    def plotlyx(self, xrange=None, opts=dict(), layout_opts=dict()):
        """
        Plot the particles' trajectory.
        Requires plotly.
        """

        import plotly.express as px
        import plotly.graph_objects as go

        d_ = self.df_for_plotting()
        if xrange:
            d_ = d_[(d_["step"] > xrange[0]) & (d_["step"] <= xrange[1])]

        pallette = px.colors.n_colors(
            "rgb(255, 0, 0)",
            "rgb(0, 125, 255)",
            len(d_["particle"].unique()),
            colortype="rgb",
        )
        # pallette = px.colors.cyclical.IceFire
        color_map = {
            p: pallette[i % len(pallette)]
            for i, p in enumerate(d_["particle"].unique())
        }

        # Plot the tokens (text) and weight (node size) for all particles and steps
        fig = px.scatter(
            d_,
            x="step",
            y="particle",
            size="prop_exp_weight",
            hover_name="token",
            color="resampled as",
            text="token",
            hover_data=["context_string", "weight", "average weight", "resample?"],
            opacity=0.15,
            color_discrete_map=color_map,
        )

        # Get resample or no-resample steps, add vline on the resample ones
        resample_steps = d_[d_["resample?"] == True]["step"].unique()
        no_resample_steps = d_[d_["resample?"] == False]["step"].unique()
        for step in resample_steps:
            fig.add_vline(x=step, line_width=4, opacity=0.15, line_color="gray")

        # Add lines connecting nodes between steps to represent resampling
        for resampled_as in d_["resampled as"].unique():
            resampled_as_data = d_[d_["resampled as"] == resampled_as]

            for step in resample_steps:
                for _, row in resampled_as_data[
                    resampled_as_data["step"] == step
                ].iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[row["step"], row["step"] + 1],
                            y=[row["resampled as"], row["particle"]],
                            mode="lines",
                            line=dict(color=color_map[resampled_as]),
                            opacity=0.3,
                            name=resampled_as,
                            showlegend=False,
                        )
                    )

            for step in no_resample_steps:
                for _, row in resampled_as_data[
                    resampled_as_data["step"] == step
                ].iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[row["step"], row["step"] + 1],
                            y=[row["particle"], row["particle"]],
                            mode="lines",
                            line=dict(color="gray"),
                            opacity=0.15,
                            name=resampled_as,
                            showlegend=False,
                        )
                    )

        # Update layout
        # fig.update_traces(textposition='middle right')
        fig.update_layout(
            plot_bgcolor="#fff",
            showlegend=False,
            margin=dict(l=0, r=0, t=15, b=20),
            xaxis=dict(showticklabels=True),
            yaxis=dict(showticklabels=False, visible=False),
        )
        return fig
