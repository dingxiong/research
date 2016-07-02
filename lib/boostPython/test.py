from personalFunctions import *
from personalPlotly import *

case = 2

if case == 1:
    x, y, z = np.random.rand(3, 10)
    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='lines+markers',
        marker=dict(
            size=12,
            line=dict(
                color='rgba(217, 217, 217, 0.14)',
                width=0.5
            ),
            opacity=0.8
        ),
        name='$\\theta$'
    )

    layout = go.Layout(
        scene=go.Scene(
            xaxis=dict(
                title='$\\theta$',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                title='$Y$',
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=go.ZAxis(title='z axis title')
        ),
        margin=dict(
            b=0,
            l=0,
            r=0,
            t=0
        ),
        showlegend=True,
    )
    fig = go.Figure(data=[trace1], layout=layout)
    py.plot(fig, filename='ex1', fileopt='overwrite')
    # plot_url = py.plot_mpl(fig, filename='ex1', fileopt='overwrite')

if case == 2:
    x, y, z = rand(3, 10)
    ptly3d(0, 1, 2, 'ex2', plotType=2, ls='dot')
    # ptly3d(x, y, z, 'ex2', plotType=2, ls='dot')
