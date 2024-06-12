from .util import format_table
import pandas as pd
import numpy as np

class SMCRecord(dict):
    """
    For storing and visualizing information about a run of SMC (an ad-hoc trace).

    Methods:
        plot(self):
            Initializes the SMCRecord instance. Accepts any positional and keyword arguments
            that are valid for the built-in dict class.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_particles = len(self['weight'][0]) # any more transparent way of getting this would smarter
    
    def __repr__(self):
        return f"SMCRecord({super().__repr__()})"
    
    def _repr_html_(self):
        # Just copied from chart.Chart  --- Might not look good.
        return ('<div>SMCRecord:<div><div style="font-family: Monospace;">'
            + format_table(self.as_df().items(), headings=['key', 'value'])
            + '</div>')
    
    def as_df(self, process=True):
        df = pd.DataFrame(self)
        if not process: return df
        df['token'] = df['context'].apply(lambda r: [ri[-1] if len(ri)>0 else [] for ri in r ])
        df['surprisal estimate'] = -1 * df['average weight'].diff() # incremental diff in avg weight (biased (over)estimator of marginalizing constant)
        return df
    
    def df_for_plotting(self):
        # Process the data for plotting. 
        recs = self.as_df(process=True).explode(['context','weight','resampled as','token']).reset_index(drop=True)
        recs['particle'] = recs.groupby(recs.index // self.n_particles).cumcount()
        recs['context_string'] = recs['context'].apply(lambda r: ''.join(r[:-1]))
        recs['exp_average weight'] = recs['average weight'].apply(lambda r: np.exp(r))
        recs['exp_weight'] = recs['weight'].apply(lambda r: np.exp(r))
        recs['prop_exp_weight'] = recs['exp_weight'] / recs['exp_average weight']
        return recs
    
    def plot_history(self, range_x=None, width=1500, height=600):
        """
        Plot the history 
        """

        import plotly.express as px
        import plotly.graph_objects as go

        recs = self.df_for_plotting()

        # Ensure color consistency by creating a color map
        pallette = px.colors.n_colors('rgb(255, 0, 0)', 'rgb(0, 125, 255)', len(recs['particle'].unique()), colortype='rgb')
        # pallette = px.colors.cyclical.IceFire
        color_map = {p: pallette[i % len(pallette)] for i, p in enumerate(recs['particle'].unique())}

        fig = px.scatter(
            recs, x="step", y="particle", size='prop_exp_weight', hover_name="token", color='resampled as', text='token',
            hover_data=["context_string","weight", "resample?"], opacity=0.2,
            # animation_frame="step",# animation_group='resampled as', 
            color_discrete_map=color_map,
            range_x=range_x,
            width=width, height=height,
        )

        resample_steps = recs[recs['resample?'] == True]['step'].unique()
        no_resample_steps = recs[recs['resample?'] == False]['step'].unique()
        for step in resample_steps:
            fig.add_vline(x=step, line_width=4, opacity=0.15, line_color="gray")

        # Add lines to represent resampling
        for resampled_as in recs['resampled as'].unique():
            resampled_as_data = recs[recs['resampled as'] == resampled_as]
            
            for step in resample_steps:
                for _, row in resampled_as_data[resampled_as_data['step'] == step].iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row['step'], row['step']+1], y=[row['resampled as'], row['particle']],
                        mode='lines', line=dict(color=color_map[resampled_as]), opacity=0.3, name=resampled_as, showlegend=False))

            for step in no_resample_steps:
                for _, row in resampled_as_data[resampled_as_data['step'] == step].iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row['step'], row['step']+1], y=[row['particle'], row['particle']], 
                        mode='lines', line=dict(color='gray'), opacity=0.2, name=resampled_as, showlegend=False))

            # for step in steps[:-1]:
            #     for _, row in resampled_as_data[resampled_as_data['step'] == step].iterrows():
            #         fig.add_trace(go.Scatter(
            #             x=[row['step']+1/2, row['step']+1], y=[row['particle'], row['particle']],
            #             mode='lines', line=dict(color='gray'), opacity=0.2, name=resampled_as, showlegend=False))
        fig.update_traces(textposition='middle right')
        fig.update_layout(plot_bgcolor="#fff", showlegend=False, )
        fig.update_yaxes(showticklabels=False, visible=False)
        fig.show()