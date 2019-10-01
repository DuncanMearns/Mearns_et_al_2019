from plotting import *
from plotting.plots.kinematics import plot_tail_kinematics
from plotting.plots.tail import generate_reconstructed_points
from datasets.main_dataset.example_data import data
import numpy as np
from matplotlib import gridspec

tinyfont = basefont.copy()
tinyfont.set_size(6)

if __name__ == "__main__":

    first_frame, last_frame = data['first_frame'], data['last_frame']
    kinematics = data['kinematics']
    example_bouts = data['example_bouts']
    convergence_threshold = data['convergence_threshold']

    # ===============
    # PLOT KINEMATICS
    # ===============

    width = 4.56
    height = 1.87

    fig = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(3, 1, left=0.04, bottom=0, right=1, top=0.92, wspace=0, hspace=0.25)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    t = np.arange(first_frame, last_frame + 1) / 500.

    # ---------
    # Plot eyes
    # ---------

    right_eye = kinematics['right'].apply(np.degrees)
    left_eye = kinematics['left'].apply(np.degrees)
    eye_max = np.ceil(max(right_eye.max(), (-left_eye).max()))
    eye_min = np.floor(min(right_eye.min(), (-left_eye).min()))
    eye_range = eye_max - eye_min

    is_converged = np.where((right_eye - left_eye) >= convergence_threshold)[0]

    ax_r = ax1
    ax_r.plot(t, right_eye, c='m', lw=2)
    ax_r.set_ylim(eye_min, eye_min + (2 * eye_range))

    ax_l = plt.twinx(ax_r)
    ax_l.plot(t, left_eye, c='g', lw=2)
    ax_l.set_ylim(-eye_min - (2 * eye_range), -eye_min)

    ax_r.fill_between(t[is_converged], -eye_min, 2 * eye_max, facecolor='0.8', lw=0)
    ax_r.text(t[is_converged][0] + 0.02, (2 * eye_max) - 12, 'Eye convergence (prey capture)', fontproperties=tinyfont,
              verticalalignment='top')

    ax_l.text(t[0] + 0.02, left_eye.iloc[0] - 5, 'L', fontproperties=tinyfont, verticalalignment='top', color='g')
    ax_r.text(t[0] + 0.02, right_eye.iloc[0] + 5, 'R', fontproperties=tinyfont, verticalalignment='bottom', color='m')

    title1 = ax1.set_title(u'Eye angle (\u00b0)', loc='left', fontproperties=verysmallfont)
    title1.set_position((0.005, 0.9))

    # Axis limits
    for ax in (ax_r, ax_l):
        ax.set_xlim(t[0], t[-1])
        open_ax(ax)
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_color('0.5')
        ax.tick_params(axis='y', which='both', color='0.5', length=3, labelcolor='0.5', pad=0.5)
        ax.tick_params(axis='y', which='minor', length=2)

    ax_l.spines['left'].set_bounds(-40, -10)
    ax_l.set_yticks([-40, -10])
    ax_l.set_yticklabels([40, 10], fontproperties=tinyfont)
    ax_l.set_yticks(np.arange(-30, -10, 10), minor=True)

    ax_r.spines['left'].set_bounds(10, 40)
    ax_r.set_yticks([10, 40])
    ax_r.set_yticklabels([10,], fontproperties=tinyfont)
    ax_r.set_yticks(np.arange(20, 40, 10), minor=True)

    # --------------------
    # Plot tail kinematics
    # --------------------
    tail_columns = [col for col in kinematics.columns if col[0] == 'k']
    tail_kinematics = kinematics.loc[:, tail_columns].applymap(np.degrees)
    plot_tail_kinematics(ax2, tail_kinematics, k_max=90)
    title2 = ax2.set_title('Rostro-caudal tail angle', loc='left', fontproperties=verysmallfont)
    title2.set_position((0.005, 0.6))

    # Create color bar
    (l, b, w, h) = ax2.get_position().bounds
    cbar = fig.add_axes((l + (w * 0.35), b + (h * 0.78), w * 0.05, h * 0.1))
    cm = plt.cm.ScalarMappable(cmap='RdBu')
    cm.set_array(np.linspace(-1, 1, 2))
    cb = fig.colorbar(cm, cax=cbar, orientation='horizontal', ticks=[])
    cb.outline.set_linewidth(0.5)
    cbar.text(-0.1, 0, u'90\u00b0(L)', fontproperties=tinyfont, horizontalalignment='right')
    cbar.text(1.1, 0, u'90\u00b0(R)', fontproperties=tinyfont, horizontalalignment='left')

    # Axis label
    ax2.text(-0.03, 0.05, 'Ro', fontproperties=tinyfont, horizontalalignment='right', verticalalignment='top')
    ax2.text(-0.03, 1, 'Ca', fontproperties=tinyfont, horizontalalignment='right', verticalalignment='bottom')

    # ---------------
    # Plot tail angle
    # ---------------
    ps = generate_reconstructed_points(kinematics.loc[:, tail_columns].values, 0)
    tip_angle = np.degrees(np.arcsin(ps[:, 1, -1] / 50.))

    ax3.plot(t, tip_angle, c='k', lw=1)
    for idx, bout_info in example_bouts.iterrows():
        t_ = np.arange(bout_info.start, bout_info.end + 1) / 500.
        ax3.fill_between(t_, np.ones((len(t_),)) * (-40), np.ones((len(t_),)) * 20, facecolor='0.8')

    # Axis limits
    open_ax(ax3)
    ax3.set_xlim(t[0], t[-1])
    ax3.set_xticks([])
    ax3.spines['bottom'].set_visible(False)

    # ax3.set_ylim(-40, 30)
    ax3.set_ylim(-50, 30)
    ax3.plot([t[-1] - 0.5, t[-1]], [-45, -45], c='k')
    ax3.set_yticks([-30, 0, 30])
    ax3.spines['left'].set_bounds(-30, 30)
    ax3.spines['left'].set_color('0.5')
    ax3.tick_params(axis='y', which='both', color='0.5', length=3, labelcolor='0.5', pad=0.5)
    ax3.set_yticklabels([-30, 0, 30], fontproperties=tinyfont)
    ax3.tick_params(axis='y', which='minor', length=2)
    ax3.set_yticks(np.arange(-20, 30, 10), minor=True)

    title3 = ax3.set_title(u'Tail tip angle (\u00b0)', loc='left', fontproperties=verysmallfont)
    title3.set_position((0.005, 0.8))

    # plt.show()
    save_fig(fig, 'figure1', 'kinematic_traces_2D')
