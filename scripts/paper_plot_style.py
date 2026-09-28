"""Shared manuscript plotting contract; experimental values are not axis ticks."""
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter

WEIGHT_TICKS = (0., .2, .4, .6, .8, 1.)
COLORS = {'oracle':'#444444', 'fit':'#b87826', 'rw':'#C05269',
          'gru':'#228b78', 'transformer':'#536eba', 'llama':'#8054A1'}
NLL_LABEL = 'Validation choice NLL (nats)'
DELTA_LABEL = 'Delta validation NLL (ablated − original)'
LINE = dict(marker='o', markersize=4, linewidth=2)
BAND_ALPHA = .10
EXPORT_DPI = 180

def apply_style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False})

def weight_axis(ax):
    ax.set_xlabel('Generating reward weight')
    ax.set_xlim(-.05, 1.05)
    ax.xaxis.set_major_locator(FixedLocator(WEIGHT_TICKS))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
