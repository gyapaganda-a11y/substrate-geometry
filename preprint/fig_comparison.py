"""
Generate a comparison figure for the preprint:
Oloid vs Cylinder CDS scores with wireframe silhouettes.
Outputs fig_comparison.pdf using matplotlib.

Run: pip install matplotlib && python fig_comparison.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={'width_ratios': [3, 2]})
fig.patch.set_facecolor('white')

# --- Left panel: CDS bar chart (log scale) ---
ax = axes[0]
names = ['Oloid\n(Schatz 1929)', 'Candidate #2\n(120/1.30/0.80)',
         'Candidate #3\n(90/0.70/0.80)', 'Candidate #1\n(120/0.70/0.80)',
         'Cylinder\n(conventional)']
scores = [8.2e-7, 8.7e-7, 1.09e-6, 1.93e-6, 4.75e-5]
colors = ['#c8a84a', '#4a9e68', '#4a9e68', '#4a9e68', '#b84a4a']

bars = ax.barh(range(len(names)), scores, color=colors, height=0.6, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels(names, fontsize=9, fontfamily='serif')
ax.set_xscale('log')
ax.set_xlabel('Contact Distribution Score (CDS) — lower is better', fontsize=10, fontfamily='serif')
ax.set_title('Rigid-Body Oracle Results', fontsize=12, fontfamily='serif', fontweight='bold', pad=12)
ax.invert_yaxis()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=8)

# Add score labels
for i, (bar, score) in enumerate(zip(bars, scores)):
    ax.text(score * 1.3, i, f'{score:.2e}', va='center', fontsize=8, fontfamily='monospace',
            color='#333' if score < 1e-5 else '#b84a4a')

# Add 58x annotation
ax.annotate('58x worse', xy=(4.75e-5, 4), xytext=(8e-6, 4.3),
            fontsize=8, fontfamily='monospace', color='#b84a4a',
            arrowprops=dict(arrowstyle='->', color='#b84a4a', lw=0.8))

# --- Right panel: Oloid wireframe sketch ---
ax2 = axes[1]
ax2.set_xlim(-1.8, 1.8)
ax2.set_ylim(-1.5, 1.5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Oloid Construction', fontsize=12, fontfamily='serif', fontweight='bold', pad=12)

# Circle 1 (XZ plane, centered at origin)
theta = np.linspace(0, 2*np.pi, 100)
r = 1.0
# Project circle 1 (in XZ plane) onto 2D with slight rotation for 3D effect
angle = 0.3  # viewing angle
c1x = r * np.cos(theta)
c1y = r * np.sin(theta) * np.cos(angle)
ax2.plot(c1x, c1y, color='#c8a84a', linewidth=1.5, alpha=0.7, label='Circle $C_1$')

# Circle 2 (YZ plane, centered at (r,0,0))
c2x = r + r * np.cos(theta) * np.sin(angle) * 0.3
c2y_raw = r * np.sin(theta)
# Shift and rotate for 3D perspective
c2x = r * np.cos(theta) * np.sin(np.pi/2) * 0.4 + 0.5
c2y = r * np.sin(theta)
ax2.plot(c2x * 0.8, c2y, color='#4a9e68', linewidth=1.5, alpha=0.7, label='Circle $C_2$')

# Convex hull outline (approximate)
hull_top = np.array([[-1, 0.2], [-0.5, 0.7], [0, 0.95], [0.5, 0.8], [1, 0.2]])
hull_bot = np.array([[-1, -0.2], [-0.5, -0.7], [0, -0.95], [0.5, -0.8], [1, -0.2]])
from scipy.interpolate import make_interp_spline
t_new = np.linspace(0, 1, 100)
spl_top = make_interp_spline(np.linspace(0, 1, len(hull_top)), hull_top, k=3)
spl_bot = make_interp_spline(np.linspace(0, 1, len(hull_bot)), hull_bot, k=3)
top_pts = spl_top(t_new)
bot_pts = spl_bot(t_new)
ax2.plot(top_pts[:,0], top_pts[:,1], color='#333', linewidth=1, alpha=0.4)
ax2.plot(bot_pts[:,0], bot_pts[:,1], color='#333', linewidth=1, alpha=0.4)

# Labels
ax2.text(0, -1.3, 'CDS = $8.2 \\times 10^{-7}$', ha='center', fontsize=10,
         fontfamily='serif', fontweight='bold', color='#c8a84a')
ax2.text(0, -1.55, 'Every surface point contacts the ground\nduring one complete roll',
         ha='center', fontsize=7, fontfamily='serif', color='#888', style='italic')
ax2.legend(fontsize=8, loc='upper right', framealpha=0.8)

plt.tight_layout(pad=2)
plt.savefig('fig_comparison.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('fig_comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved fig_comparison.pdf and fig_comparison.png")
