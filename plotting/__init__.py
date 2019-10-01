from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib.font_manager import FontProperties
import os


# Figure settings
rcParams['pdf.fonttype'] = 42
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['axes.unicode_minus'] = False


# Saving
output_directory = 'D:\\PAPER\\final_figures'
def save_fig(fig, folder_name, file_name, ext='pdf'):
    global output_directory
    subdir = os.path.join(output_directory, str(folder_name))
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    fig.savefig(os.path.join(subdir, '.'.join([file_name, ext])))


# Fonts
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = 'Arial'
basefont = FontProperties()
largefont = basefont.copy()
largefont.set_size(12)
smallfont = basefont.copy()
smallfont.set_size(10)
verysmallfont = basefont.copy()
verysmallfont.set_size(8)


# Plotting helpers
def open_ax(ax=None):
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
