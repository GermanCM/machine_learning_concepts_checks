#%%
from datetime import datetime

#%%
class Dataset_plots():
    """
        This class is ready to: 
            - receive an existing 'dataframe' 
            - build it from a csv file
            - build it from a Onesait ontology given the necessary data  
    """
    def __init__(self, dataframe=None):
        self.dataframe = dataframe
        
    def plot_series_values(self, dataframe, attribute, figsize_rows=15, figsize_cols=10):
        try:
            import matplotlib.pyplot as plt
            plt.subplots(figsize=(figsize_rows, figsize_rows))  
            plt.plot(dataframe[attribute].values)  

            plt.show()
        
        except Exception as exc:
            #logger.exception('raised exception at {}: {}'.format(logger.name+'.'+self.plot_series_values.__name__, exc))
            raise exc

    def make_line_plots(self, title, dataset, attributes_to_plot=[], x_axis_label='x', y_axis_label='y', 
                        background_color=None, legend_location="top_left", line_colors=None):
        try:
            from bokeh.plotting import figure, show, output_file
            from bokeh.models import HoverTool
            from bokeh.io import output_notebook

            output_notebook()

            p = figure(title=title, tools='', background_fill_color=background_color)
            p.legend.location = legend_location

            x_values = dataset.index
            i = 0
            for attribute in attributes_to_plot:
                    attribute_values = dataset[attribute]
                    p.line(x_values, attribute_values, legend=attribute, line_dash=[4, 4], line_color='red', #line_colors[i], 
                            line_width=2)
                    i += 1

            p.y_range.start = 0
            #p.legend.location = "center_right"
            p.legend.background_fill_color = background_color
            p.xaxis.axis_label = x_axis_label
            p.yaxis.axis_label = y_axis_label
            p.grid.grid_line_color="white"

            p.add_tools(HoverTool())
            p.legend.click_policy="hide"

            show(p)

        except Exception as exc:
            raise exc

    def plot_series_histogram(self, attribute, figsize_rows=15, figsize_cols=10):
        try:
            import matplotlib.pyplot as plt
            plt.subplots(figsize=(figsize_rows, figsize_rows))
            plt.hist(self.dataframe[attribute])

            plt.show()
        
        except Exception as exc:
            raise exc

    def make_plot(self, title, hist, edges, x, pdf, cdf, x_axis_label='x', y_axis_label='Pr(x)'):
        try:
            from bokeh.plotting import figure
            from bokeh.models import HoverTool

            p = figure(title=title, tools='', background_fill_color="#fafafa")
            p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color="navy", line_color="white", alpha=0.5)
            p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend="PDF")
            p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend="CDF")

            p.y_range.start = 0
            p.legend.location = "center_right"
            p.legend.background_fill_color = "#fefefe"
            p.xaxis.axis_label = x_axis_label
            p.yaxis.axis_label = y_axis_label
            p.grid.grid_line_color="white"

            p.add_tools(HoverTool())
            p.legend.click_policy="hide"

            return p
        except Exception as exc:
            raise exc

    '''
    Take into account that numpy histogram function needs no nan values; you can apply a mask to your values beforehand:
    mask_not_nan_values = np.isnan(values)==False
    values = values[mask_not_nan_values]
    '''
    def plot_Hist_Pdf_Cdf(self, values, mu, sigma, graph_title=None, x_axis_label='x', y_axis_label='Pr(x)'):
        try:
            import numpy as np
            import scipy as scp
            import scipy.special   
            from bokeh.layouts import gridplot
            from bokeh.plotting import figure, show, output_file

            x = np.linspace(0, np.max(values), 1000)
            pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
            cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2
            graph_title = graph_title
            hist, edges = np.histogram(values, density=True, bins=100)
            p1 = make_plot(graph_title, hist, edges, x, pdf, cdf, x_axis_label, y_axis_label)

            #output_file('histogram.html', title="histogram.py example")
            show(gridplot([p1], ncols=2, plot_width=400, plot_height=400, toolbar_location=None))
            return
        
        except Exception as exc:
            raise exc

    def plot_QQ_plot(self, series_values):
        try:
            from numpy.random import seed
            from numpy.random import randn
            from statsmodels.graphics.gofplots import qqplot
            from matplotlib import pyplot
            
            qqplot(series_values, line='s')
            pyplot.show()
        except Exception as exc:
            raise exc