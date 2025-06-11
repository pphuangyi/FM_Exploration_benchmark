"""
Functions to plot ROC and PR curves
"""
import plotly.graph_objects as go

def plot_distribution(data, 
                      labels, 
                      data_type = None, 
                      vmin      = 0, 
                      vmax      = 1, 
                      bin_size  = 0.002, 
                      log_y     = True):
    fig = go.Figure()
    
    data_same = data[labels].detach().cpu().numpy()
    data_diff = data[~labels].detach().cpu().numpy()

    # by definition the maximum distance is 1
    xbins=dict(start=vmin, end=vmax, size=bin_size)

    # distance of different-track pairs
    trace = go.Histogram(
        x=data_same,
        name=f'same-track {data_type}' if data_type is not None else 'distribution',
        opacity=0.4,
        marker_color='deepskyblue',
        histnorm='probability density',
        xbins=xbins,
    )
    fig.add_trace(trace)

    # distance of different-track pairs
    trace = go.Histogram(
        x=data_diff,
        name=f'diff-track {data_type}' if data_type is not None else 'distribution',
        opacity=0.4,
        marker_color='tomato',
        histnorm='probability density',
        xbins=xbins,
    )
    fig.add_trace(trace)
    
    fig.update_layout(
        barmode='overlay',
        title=f'Distribution of {data_type} of same- and diff- track pairs',
        xaxis=dict(title='distance'),
        yaxis=dict(title='Density', type='log' if log_y else 'linear'),
        template='plotly_white'
    )
    
    fig.show()

def plot_roc_curve(fpr, tpr, auc):
    """
    Plot ROC curve
    """
    fig = go.Figure()

    # Add ROC curve
    trace = go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines+markers',
        name='ROC Curve',
        line=dict(color='blue'),
        hovertemplate='FPR: %{x:.2f}<br>TPR: %{y:.2f}<extra></extra>'
    )
    fig.add_trace(trace)

    # Reference line
    trace = go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', dash='dash')
    )
    fig.add_trace(trace)

    fig.update_layout(
        title=f'ROC Curve (AUC = {auc:.4f})',
        xaxis_title='False Positive Rate (FPR)',
        yaxis_title='True Positive Rate (TPR)',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=700,
        height=500,
        template='plotly_white'
    )

    fig.show()


def plot_pr_curve(recall, precision, average_precision):
    """
    Plot the PR curve
    """
    fig = go.Figure()

    # Add PR curve
    trace = go.Scatter(
        x=recall,
        y=precision,
        mode='lines+markers',
        name='PR Curve',
        line=dict(color='green'),
        hovertemplate='Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra></extra>'
    )
    fig.add_trace(trace)

    fig.update_layout(
        title=f'Precision-Recall Curve (AP = {average_precision:.4f})',
        xaxis_title='Recall',
        yaxis_title='Precision',
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        width=700,
        height=500,
        template='plotly_white'
    )

    fig.show()