import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

models_data = {
    "Student Auxiliary": {"acc": 98.48, "params": 847618,  "flops": 673414720, "color": "#FFB7B2"},
    "Student Main":      {"acc": 98.56, "params": 1065346,  "flops": 684855328, "color": "#FF9AA2"},
    "Teacher Auxiliary": {"acc": 98.48, "params": 2968642,  "flops": 1421171968, "color": "#A2D2FF"},
    "Teacher Main":      {"acc": 98.65, "params": 11177538, "flops": 1823522816, "color": "#89CFF0"},
    "MobileNetV2":       {"acc": 98.02, "params": 2226434,  "flops": 319021570, "color": "#B5EAD7"},
    "EfficientNetB0":    {"acc": 98.52, "params": 4010110,  "flops": 408918458, "color": "#E2F0CB"},
    "DenseNet121":       {"acc": 98.55, "params": 6955906,  "flops": 2898820098, "color": "#C7CEEA"},
    "ResNet18":          {"acc": 98.50, "params": 11177538, "flops": 1824801794, "color": "#D7BDE2"},
    "Swin-Tiny":         {"acc": 98.38, "params": 27520892, "flops": 3127415954, "color": "#F0B27A"},
    "ViT-Base":          {"acc": 97.83, "params": 85800194, "flops": 17605263122, "color": "#808B96"},
    "ConvNeXt-Base":     {"acc": 98.27, "params": 87568514, "flops": 15424556034, "color": "#95A5A6"},
    "VGG16":             {"acc": 97.87, "params": 134268738,"flops": 15519136770, "color": "#566573"}
}

def convert_params(n): return round(n / 1e6, 2)
def convert_flops(n): return round(n / 1e9, 2)

# Split data
names, accs, params, flops = [], [], [], []
for name, data in models_data.items():
    names.append(name)
    accs.append(data['acc'])
    params.append(convert_params(data['params']))
    flops.append(convert_flops(data['flops']))

names = names[::-1]; accs = accs[::-1]; params = params[::-1]; flops = flops[::-1]
y_pos = np.arange(len(names))

# --- 2. Setup colors ---
target_color = '#1680b4'

# --- 3. Setup SUBPLOTS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8), sharey=True,
                               gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.05})

# ==========================================
# Column 1: ACCURACY & GFLOPs
# ==========================================

xlim_left, xlim_right = 97.0, 99.5
ax1.set_xlim(xlim_left, xlim_right)

# 2. Plot Main Bar (Accuracy)
bars = ax1.barh(y_pos, width=[a - xlim_left for a in accs], left=xlim_left,
                color=target_color, height=0.65, label='Accuracy')

# 3. Text Accuracy
for i, bar in enumerate(bars):
    width = bar.get_width()
    text_color = 'white'
    ax1.text(xlim_left + width - 0.05, bar.get_y() + bar.get_height()/2,
             f"{accs[i]:.2f}%", va='center', ha='right', color=text_color, fontweight='bold', fontsize=10)

# 4. Plot Line Chart (GFLOPs)
ax1_twin = ax1.twiny()
ax1_twin.set_xlim(min(flops)*0.5, max(flops)*1.1)

line_color = '#0c0e0d'

ax1_twin.plot(flops, y_pos, linestyle='--', linewidth=2, color=line_color,
              marker='*', markersize=14, markerfacecolor=line_color, markeredgecolor='white')

# Text GFLOPs
for i, (fl, y) in enumerate(zip(flops, y_pos)):
    if names[i] == "Swin-Tiny":
        ha_align = 'right'
        offset = -0.5
    else:
        ha_align = 'left'
        offset = 0.5

    ax1_twin.text(fl + offset, y, f"{fl}", va='center', ha=ha_align,
                  color=line_color, fontsize=9, fontweight='bold')

# Style column 1
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(2); ax1.spines['left'].set_color('black')
ax1.spines['bottom'].set_linewidth(1.5); ax1.spines['bottom'].set_color('black')

ax1.set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold', labelpad=10, color='black')
ax1.tick_params(axis='x', labelsize=11, colors='black')

ax1_twin.spines['top'].set_visible(False); ax1_twin.spines['right'].set_visible(False)
ax1_twin.spines['left'].set_visible(False); ax1_twin.spines['bottom'].set_visible(False)
ax1_twin.set_xticks([])
ax1_twin.text(flops[-1] + 0.5, y_pos[-1] + 0.4, 'GFLOPs', color=line_color, fontsize=11, fontweight='bold', ha='center')

ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, fontsize=11, fontweight='bold', color='black')

# ==========================================
# Column 2: PARAMS
# ==========================================

ax2.barh(y_pos, width=params, color='#95a5a6', height=0.65, alpha=0.8)
ax2.set_xlim(0, max(params) * 1.35)

for i, v in enumerate(params):
    ax2.text(v + 1, y_pos[i], f"{v}M", va='center', ha='left', color='#1B2631', fontsize=10, fontstyle='italic', fontweight='bold')

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_linewidth(1.5); ax2.spines['bottom'].set_color('black')
ax2.tick_params(left=False, labelleft=False)

ax2.set_xlabel('Parameters (M)', fontsize=12, fontweight='bold', labelpad=10, color='black')
ax2.tick_params(axis='x', labelsize=11, colors='black')

plt.tight_layout()
plt.savefig(r"bar_plot_acc_params_flops.jpg", dpi=300, bbox_inches="tight")
plt.show()