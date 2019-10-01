from .colormap import ColorMap

PuOr = ColorMap('PuOr', (-1, 1))
colors = PuOr.map([-1, -0.5, 1, 0.5])
mutant_colors = {'blu_s257': {'het': colors[0], 'mut': colors[1]},
                 'lakritz': {'ctrl': colors[2], 'mut': colors[3]}}
