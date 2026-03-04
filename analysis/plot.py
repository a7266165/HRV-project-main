import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_taichi_plot(lf_hf_ratio, lf_nu=None, hf_nu=None, add_legend=True):
    
    total = 1 + lf_hf_ratio
    sympathetic_ratio = lf_hf_ratio / total    
    parasympathetic_ratio = 1 / total         
    
    sympathetic_ratio = 1/total
    parasympathetic_ratio = lf_hf_ratio/total

    fig, ax = plt.subplots(figsize=(6, 6))
    
    R = 1.0  
    r_sympathetic = R * sympathetic_ratio
    r_parasympathetic = R * parasympathetic_ratio
    
    
    #白底
    base = patches.Circle((0, 0), R, facecolor='white', edgecolor='none')
    ax.add_patch(base)
    
    #右半黑色
    right_half = patches.Wedge((0, 0), R, -90, 90, 
                               facecolor='black', edgecolor='none', linewidth=0)
    ax.add_patch(right_half)
    
    #下方黑色
    bottom_circle = patches.Circle((0, -R + r_sympathetic), r_sympathetic, 
                                  facecolor='black', edgecolor='none', linewidth=0)
    ax.add_patch(bottom_circle)
    
    #上方白色
    top_circle = patches.Circle((0, R - r_parasympathetic), r_parasympathetic, 
                               facecolor='white', edgecolor='none', linewidth=0)
    ax.add_patch(top_circle)
    
    #最外框
    outer_circle = patches.Circle((0, 0), R, fill=False, 
                                 edgecolor='black', linewidth=2)
    ax.add_patch(outer_circle)

    if add_legend:
        legend_x = 1.3
        ax.scatter(legend_x, 0.3, c='black', s=100, marker='o')
        ax.text(legend_x + 0.1, 0.3, '交感', fontsize=12, va='center')
        ax.scatter(legend_x, -0.3, c='white', s=100, marker='o', 
                  edgecolors='black', linewidth=1)
        ax.text(legend_x + 0.1, -0.3, '副交感', fontsize=12, va='center')
    
    if lf_nu is not None and hf_nu is not None:
        info_text = f'LF/HF ratio = {lf_hf_ratio:.3f}\nLFnu = {lf_nu:.1f}%\nHFnu = {hf_nu:.1f}%'
    else:
        info_text = f'LF/HF ratio = {lf_hf_ratio:.3f}\n交感: {sympathetic_ratio*100:.1f}%\n副交感: {parasympathetic_ratio*100:.1f}%'
    
    ax.text(0, -1.4, info_text, ha='center', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    ax.set_xlim(-1.2, 1.6)
    ax.set_ylim(-1.6, 1.2) 
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

