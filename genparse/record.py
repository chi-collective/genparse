import numpy as np
import pandas as pd
import copy


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
        self.n_particles = self['n_particles']
        self.algorithm = self['algorithm']
        self.history = self['history']

    def __repr__(self):
        return f'SMCRecord({super().__repr__()})'

    def _repr_html_(self):
        # Just copied from chart.Chart  --- Might not look good.
        return (
            f'<div>SMCRecord: '
            f'n_particles={self.n_particles}, algorithm={self.algorithm}</div>'
            f'<div>History:' + self.to_df().to_html() + '</div>'
        )

    def reorganize_history(self, untangle=False):
        """
        Reorganize the history data into a list of dictionaries, for processing into a pandas dataframe.
        """
        output_data = []

        history = self.history if not untangle else untangle_history(self.history)

        for step_data in history:
            step = step_data['step']
            particles = step_data['particles']

            context = [p['context'] for p in particles]
            weight = [p['weight'] for p in particles]

            # Assuming all particles have the same context length
            token = [p['context'][-1] if p['context'] else '' for p in particles]

            new_entry = {
                'step': step,
                'context': context,
                'weight': weight,
                'resample_indices': step_data.get(
                    'resample_indices', list(range(len(particles)))
                ),
                'resample?': 'resample_indices' in step_data,
                'average weight': step_data['average_weight'],
                'token': token,
                'change in w': step_data['average_weight']
                if step == 1
                else step_data['average_weight'] - output_data[-1]['average weight'],
            }

            output_data.append(new_entry)

        return output_data

    def to_df(self, process=True, untangle=False):
        df = pd.DataFrame(self.reorganize_history(untangle=untangle))
        if not process:
            return df
        df['token'] = df['context'].apply(
            lambda x: [ri[-1] if len(ri) > 0 else [] for ri in x]
        )
        df['change in w'] = df[
            'average weight'
        ].diff()  # incremental diff in avg weight (biased (over)estimator of marginalizing constant)
        return df

    def df_for_plotly(self, untangle=False):
        # Process the data for plotting.
        recs = (
            self.to_df(process=True, untangle=untangle)
            .explode(['context', 'weight', 'resample_indices', 'token'])
            .reset_index(drop=True)
        )
        recs['particle'] = recs.groupby(recs.index // self.n_particles).cumcount()
        recs['context_string'] = recs['context'].apply(lambda x: ''.join(x[:-1]))
        recs['exp_average weight'] = recs['average weight'].apply(np.exp)
        recs['exp_weight'] = recs['weight'].apply(lambda x: np.exp(x))  # pylint: disable=unnecessary-lambda
        recs['prop_exp_weight'] = recs['exp_weight'] / recs['exp_average weight']
        return recs

    def plotly(
        self, show_est_logprob=False, xrange=None, height=None, width=None, untangle=True
    ):
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        d_ = self.df_for_plotly(untangle=untangle)

        if xrange:
            d_ = d_[(d_['step'] > xrange[0]) & (d_['step'] <= xrange[1])]

        # Define color palette
        pallette = px.colors.n_colors(
            'rgb(255, 0, 0)',
            'rgb(0, 125, 255)',
            len(d_['particle'].unique()),
            colortype='rgb',
        )
        color_map = {
            p: pallette[i % len(pallette)] for i, p in enumerate(d_['particle'].unique())
        }

        # Initialize the figure
        fig = make_subplots(
            rows=2 if show_est_logprob else 1,
            cols=1,
            row_heights=[0.8, 0.2] if show_est_logprob else None,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('particles', 'est. logprob (change in average weight)'),
        )

        # Add scatter plot points
        for resampled_as in d_['resample_indices'].unique():
            resampled_as_data = d_[d_['resample_indices'] == resampled_as]
            fig.add_trace(
                go.Scatter(
                    x=resampled_as_data['step'],
                    y=resampled_as_data['particle'],
                    mode='markers+text',
                    marker=dict(
                        size=np.sqrt(resampled_as_data['prop_exp_weight'] * 200),
                        color=color_map[resampled_as],
                        opacity=0.15,
                    ),
                    text=resampled_as_data['token'],
                    hoverinfo='text',
                    hovertext=resampled_as_data.apply(
                        lambda row: (
                            f"Token:    {'`<b>'+row['token']+'</b>`' if row['token'] else ''}<br>"
                            + f"Context:  {row['context_string']}<br>"
                            + f"Step {row['step']}; Avg weight = {row['average weight']:4f}<br>"
                            + f"Particle {row['particle']}; Weight = {row['weight']:4f}<br>"
                            + f"{'        ↳ resample_indices particle '+str(row['resample_indices']) if row['resample?'] else ''}"
                        ),
                        axis=1,
                    ),
                    showlegend=False,
                )
            )

        # Add lines connecting nodes between steps to represent resampling
        for resampled_as in d_['resample_indices'].unique():
            resampled_as_data = d_[d_['resample_indices'] == resampled_as]
            for step in d_['step'].unique():
                for _, row in resampled_as_data[
                    resampled_as_data['step'] == step
                ].iterrows():
                    fig.add_trace(
                        go.Scatter(
                            x=[row['step'], row['step'] + 1],
                            y=[row['resample_indices'], row['particle']],
                            mode='lines',
                            line=dict(color=color_map[resampled_as]),
                            opacity=0.2,
                            name=resampled_as,
                            showlegend=False,
                        )
                    )

        if show_est_logprob:
            fig.append_trace(
                go.Scatter(
                    x=d_['step'],
                    y=d_['change in w'],
                    line=dict(color='green'),
                    mode='lines+markers',
                    hoverinfo='text',
                    hovertext=d_.apply(lambda row: row['change in w'], axis=1),
                ),
                row=2,
                col=1,
            )

            # # Get resample or no-resample steps, add vline on the resample ones
            for step in d_[d_['resample?']]['step'].unique():
                fig.add_vline(
                    x=step, line_width=4, opacity=0.15, line_color='gray', row=2
                )

        # Update layout to include range slider

        # fig.update_traces(textposition='middle right')
        fig.update_layout(
            plot_bgcolor='#fff',
            showlegend=False,
            margin=dict(l=10, r=10, t=40, b=10),
            height=height,
            width=width,
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


def untangle_history(history):
    untangled_history = []

    # to avoid modifying the original history in place
    history_copy = copy.deepcopy(history)

    for t, step in enumerate(history_copy):
        if t == 0:
            untangled_history.append(step)
            continue

        prev_step = untangled_history[t - 1]

        if 'resample_indices' in prev_step:
            # Create a permutation based on the previous step's resample_indices
            perm = create_permutation(prev_step['resample_indices'])

            # Reorder previous step's resample_indices
            prev_step['resample_indices'] = perm['permute'](prev_step['resample_indices'])

            # Reorder current step's particles
            step['particles'] = perm['permute'](step['particles'])

            # Update current step's resample_indices if it exists
            if 'resample_indices' in step:
                step['resample_indices'] = perm['re_index'](step['resample_indices'])

        # reorder the resample_indices if they're there and it's the last step
        if t == len(history) - 1 and 'resample_indices' in step:
            perm_current = create_permutation(step['resample_indices'])
            step['resample_indices'] = perm_current['permute'](step['resample_indices'])

        untangled_history.append(step)

    return untangled_history


def create_permutation(arr):
    from_index = sorted(range(len(arr)), key=lambda i: arr[i])
    to_index = [from_index.index(i) for i in range(len(arr))]

    def permute(array):
        return [array[i] for i in from_index]

    def re_index(indices):
        return [to_index[i] for i in indices]

    return {'permute': permute, 're_index': re_index}
