import plotly.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go


def ptlyTrace3d(x, y, z, plotType=0, lw=2, lc='blue', ls='solid',
                mc='red', ms=5, mt='circle', label=None):
    
    if plotType == 0:
        mode = 'lines'
    elif plotType == 1:
        mode = 'markers'
    else:
        mode = 'lines' + 'markers'

    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode=mode,
        marker=dict(
            size=ms,
            color=mc,
            symbol=mt,  # "circle" | "square" | "diamond" | "cross" | "x"
        ),
        line=dict(
            color=lc,
            width=lw,
            dash=ls,            # solid, dot, dash
        ),
        name=label
    )
    return trace


def ptlyLayout3d(labs=['$x$', '$y$', '$z$'], showLegend=False):
    layout = go.Layout(
        scene=go.Scene(
            xaxis=dict(
                title=labs[0],
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            yaxis=dict(
                title=labs[1],
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            ),
            zaxis=dict(
                title=labs[2],
                showbackground=True,
                backgroundcolor='rgb(230, 230,230)'
            )
        ),
        margin=dict(
            b=0,
            l=0,
            r=0,
            t=0
        ),
        showlegend=showLegend,
    )

    return layout


def ptly3d(trace, fileName, off=True, labs=['$x$', '$y$', '$z$'],
           showLegend=False):
    layout = ptlyLayout3d(labs=labs, showLegend=showLegend)
    fig = go.Figure(data=trace, layout=layout)
    if off:
        pyoff.plot(fig, filename=fileName)
    else:
        py.plot(fig, filename=fileName, fileopt='overwrite')


def ptlyPlot3d(x, y, z, fileName, off=True, plotType=0, lw=2, lc='blue',
               ls='solid',
               mc='red', ms=5,  mt='circle', label=None,
               labs=['$x$', '$y$', '$z$'], showLegend=False):
    trace = ptlyTrace3d(x, y, z, plotType=plotType, lw=lw, lc=lc, ls=ls,
                        mc=mc, ms=ms, mt=mt, label=label)
    ptly3d([trace], fileName, off=off, labs=labs, showLegend=showLegend)


