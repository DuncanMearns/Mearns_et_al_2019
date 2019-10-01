from colormap import ColorMap

lensmap = ColorMap('BrBG', (-1, 1))
lensectomy_colors = {}
lensectomy_colors['control'] = lensmap.map(0.6)
lensectomy_colors['unilateral'] = lensmap.map(-0.6)
lensectomy_colors['left'] = lensmap.map(-0.75)
lensectomy_colors['right'] = lensmap.map(-0.75)
lensectomy_colors['bilateral'] = (0.5, 0.5, 0.5, 1.0)
