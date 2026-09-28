"""A readable two-question view; retain the complete existing diagnostic figure."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from complete_task1_diagnostics import OUT, WEIGHTS
from paper_plot_style import apply_style, weight_axis, COLORS


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    apply_style()
    colors = dict(COLORS)
    families = [('gru', 'GRU'), ('transformer', 'Transformer')]
    has_llama = 'llama_choice_nll' in summary[str(WEIGHTS[0])]['all']
    if has_llama:
        colors['llama'] = '#A34D79'
        families.append(('llama', 'LLaMA'))

    def curve(ax, key, label, color, style='-', band=True):
        rows = [summary[str(w)]['all'][key] for w in WEIGHTS]
        ax.plot(WEIGHTS, [r['mean'] for r in rows], style, color=color,
                marker='o', ms=4, lw=2, label=label)
        if band:
            ci = np.array([r['ci95'] for r in rows])
            ax.fill_between(WEIGHTS, ci[:, 0], ci[:, 1], color=color, alpha=.12)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    for f, label in families:
        curve(axs[0], f + '_choice_nll', label + ' choice-only', colors[f])
        curve(axs[1], f + '_reward_predictive_gain', label, colors[f])
    curve(axs[0], 'oracle_nll', 'Oracle (with reward)',
          '#454545', '--', band=False)
    axs[0].axhline(np.log(4), color='#aaaaaa', ls=':', lw=1.2)
    axs[0].text(.98, np.log(4) - .025, 'Uniform random', ha='right', va='top',
                color='#777777', fontsize=10)
    axs[0].set(title='a  Prediction from choices alone',
               ylabel='Choice NLL (lower is better)', ylim=(.3, 1.45))
    axs[1].axhline(0, color='#777777', lw=1)
    axs[1].set(title='b  Predictive benefit of reward',
               ylabel='NLL(choice-only) − NLL(full input)', ylim=(-.09, .14 if has_llama else .1))
    axs[1].text(.04, .133 if has_llama else .093, 'Above zero: full input predicts better',
                fontsize=10, va='top', color='#555555')
    axs[1].text(.04, -.084, 'Below zero: choice-only predicts better',
                fontsize=10, va='bottom', color='#555555')
    for ax in axs:
        weight_axis(ax)
        ax.grid(axis='y', alpha=.15)
        ax.legend(loc='upper right', frameon=False, fontsize=10,
                  bbox_to_anchor=(1, .29) if ax is axs[1] else (1, .87))
    fig.suptitle('A | Predictive information with and without reward',
                 fontsize=16, x=.08, ha='left', y=.985)
    fig.subplots_adjust(left=.08, right=.98, top=.84, bottom=.23, wspace=.32)
    fig.text(.08, .055,
             ('All participants · Validation trials 151–200 · 250 participants/weight · GRU/Transformer: 3 seeds; LLaMA: 1 seed\n' if has_llama else 'All participants · Validation trials 151–200 · 250 participants/weight · 3 seeds/model\n') +
             'Bands: paired participant bootstrap 95% CI, conditional on seeds. Validation also used for early stopping.\n'
             'Oracle observes rewards; choice-only networks do not. Behavioral baselines are in the supplement. No test access.',
             fontsize=9, color='#555555')
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'section_a_main.{ext}', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
