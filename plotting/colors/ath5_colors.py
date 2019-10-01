from .colormap import ColorMap

Winter = ColorMap('winter', (-1, 1))
colors = Winter.map([-1, 1])
ath5_colors = {'control': colors[0], 'ablated': colors[1]}
