from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_run(run_path):
    p=Path(run_path);h=json.loads((p/'history.json').read_text())
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':9,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(figsize=(3.4,2.4));ax.plot([v['epoch'] for v in h],[v['validation_balanced_accuracy'] for v in h],color='#376b7e',marker='o',lw=1.2,ms=3)
    ax.set(xlabel='Training epoch',ylabel='Validation balanced accuracy',ylim=(0,1));ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(p/f'validation_curve.{ext}',dpi=600)
    plt.close(fig)
    (p/'figure_notes.md').write_text('Validation balanced accuracy by epoch. Check provenance.json for task, cohort and smoke status. Reused validation observations are not independent test results.\n')
    return p/'validation_curve.png'
