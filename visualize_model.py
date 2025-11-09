"""
Model Architecture Visualization Script
Generates visual representations of the YOLO custom model architecture
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from yolo_custom import YOLOCustom, count_parameters


def visualize_architecture():
    """Create a horizontal left-to-right diagram of the model architecture"""
    
    # Model layer information with connections
    layers_info = [
        # idx, name, channels, params, resolution, from_layer, y_offset
        (0, "Conv", "3→32", "928", "320²", -1, 0),
        (1, "Conv", "32→64", "18.6K", "160²", 0, 0),
        (2, "C3k2", "64→128", "23.0K", "160²", 1, 0),
        (3, "Conv", "128→128", "147.7K", "80²", 2, 0),
        (4, "C3k2", "128→256", "91.1K", "80²", 3, 0),
        (5, "Conv", "256→256", "590.3K", "40²", 4, 0),
        (6, "C3k2", "256→256", "296.4K", "40²", 5, 0),
        (7, "Conv", "256→512", "1.18M", "20²", 6, 0),
        (8, "C3k2", "512→512", "1.18M", "20²", 7, 0),
        (9, "SPPF", "512→512", "656.9K", "20²", 8, 0),
        (10, "C2PSA", "512→512", "1.31M", "20²", 9, 0),
        (11, "Up×2", "512", "0", "40²", 10, 1),
        (12, "Concat", "[11,6]→768", "0", "40²", [11, 6], 0.5),
        (13, "C3k2", "768→256", "427.5K", "40²", 12, 0),
        (14, "Up×2", "256", "0", "80²", 13, 1),
        (15, "Concat", "[14,4]→512", "0", "80²", [14, 4], 0.5),
        (16, "C3k2", "512→128", "123.6K", "80²", 15, 0),
        (17, "Conv", "128→128", "147.7K", "40²", 16, -1),
        (18, "Concat", "[17,13]→384", "0", "40²", [17, 13], -0.5),
        (19, "C3k2", "384→256", "329.2K", "40²", 18, 0),
        (20, "Conv", "256→256", "590.3K", "20²", 19, -1),
        (21, "Concat", "[20,10]→768", "0", "20²", [20, 10], -0.5),
        (22, "C3k2", "768→512", "1.31M", "20²", 21, 0),
        (23, "Segment", "Multi-out", "423.4K", "Multi", [16, 19, 22], 0),
    ]
    
    fig, ax = plt.subplots(figsize=(28, 12))
    ax.set_xlim(-1, 25)
    ax.set_ylim(-4, 4)
    ax.axis('off')
    
    # Title
    ax.text(12, 3.5, 'YOLO Custom Segmentation Architecture (Left→Right Flow)', 
            ha='center', fontsize=18, fontweight='bold')
    
    # Color scheme
    colors = {
        'Conv': '#FF6B6B',
        'C3k2': '#4ECDC4',
        'SPPF': '#95E1D3',
        'C2PSA': '#F38181',
        'Up×2': '#FFA07A',
        'Concat': '#DDA15E',
        'Segment': '#BC6C25'
    }
    
    # Position tracking
    positions = {}
    x_spacing = 1.0
    box_width = 0.8
    box_height = 1.2
    
    # Draw all layers
    for idx, name, channels, params, resolution, from_layer, y_offset in layers_info:
        x_pos = idx * x_spacing
        y_pos = y_offset
        
        color = colors.get(name, '#CCCCCC')
        
        # Main box
        box = FancyBboxPatch((x_pos - box_width/2, y_pos - box_height/2), 
                             box_width, box_height,
                             boxstyle="round,pad=0.05", 
                             edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(box)
        
        # Store position
        positions[idx] = (x_pos, y_pos)
        
        # Layer index (top)
        ax.text(x_pos, y_pos + 0.45, f'{idx}',
                ha='center', va='center', fontweight='bold', fontsize=9,
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=1))
        
        # Layer name (center)
        ax.text(x_pos, y_pos + 0.1, name,
                ha='center', va='center', fontweight='bold', fontsize=9)
        
        # Channels (below name)
        ax.text(x_pos, y_pos - 0.15, channels,
                ha='center', va='center', fontsize=7, style='italic')
        
        # Parameters (bottom)
        ax.text(x_pos, y_pos - 0.35, params,
                ha='center', va='center', fontsize=7, color='darkred', fontweight='bold')
        
        # Resolution (very bottom)
        ax.text(x_pos, y_pos - 0.55, resolution,
                ha='center', va='center', fontsize=6, color='darkblue')
    
    # Draw connections
    for idx, name, channels, params, resolution, from_layer, y_offset in layers_info:
        if idx == 0:
            continue
            
        x_pos, y_pos = positions[idx]
        
        if isinstance(from_layer, list):
            # Multiple inputs (Concat or Segment)
            for src_idx in from_layer:
                src_x, src_y = positions[src_idx]
                
                # Different line styles for skip connections
                if abs(src_idx - idx) > 2:
                    # Skip connection (dashed)
                    linestyle = '--'
                    linewidth = 2
                    alpha = 0.7
                    color = 'blue' if src_y > y_pos else 'green'
                else:
                    # Local connection
                    linestyle = '-'
                    linewidth = 2
                    alpha = 0.9
                    color = 'black'
                
                # Create curved arrow
                arrow = FancyArrowPatch((src_x + box_width/2, src_y), 
                                       (x_pos - box_width/2, y_pos),
                                       arrowstyle='->', mutation_scale=15,
                                       linewidth=linewidth, color=color,
                                       linestyle=linestyle, alpha=alpha,
                                       connectionstyle="arc3,rad=0.3")
                ax.add_patch(arrow)
        else:
            # Single input
            src_x, src_y = positions[from_layer]
            
            # Straight or curved based on y difference
            if abs(src_y - y_pos) < 0.1:
                # Straight horizontal arrow
                arrow = FancyArrowPatch((src_x + box_width/2, src_y), 
                                       (x_pos - box_width/2, y_pos),
                                       arrowstyle='->', mutation_scale=15,
                                       linewidth=2.5, color='black')
            else:
                # Curved arrow for vertical offset
                arrow = FancyArrowPatch((src_x + box_width/2, src_y), 
                                       (x_pos - box_width/2, y_pos),
                                       arrowstyle='->', mutation_scale=15,
                                       linewidth=2, color='black',
                                       connectionstyle="arc3,rad=0.2")
            ax.add_patch(arrow)
    
    # Add section labels
    ax.text(5, -2.8, 'BACKBONE', fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE5E5', edgecolor='red', linewidth=2))
    ax.text(16, -2.8, 'NECK', fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#E5F5FF', edgecolor='blue', linewidth=2))
    ax.text(23, -2.8, 'HEAD', fontsize=13, fontweight='bold', ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF5E5', edgecolor='orange', linewidth=2))
    
    # Legend with detailed explanations
    legend_y = 2.5
    legend_x = 0.5
    
    # Title for legend
    ax.text(legend_x, legend_y + 0.3, 'Legend:', fontsize=11, fontweight='bold')
    
    legend_items = [
        (colors['Conv'], 'Conv', 'Convolution + BN + SiLU'),
        (colors['C3k2'], 'C3k2', 'CSP Bottleneck Block'),
        (colors['SPPF'], 'SPPF', 'Spatial Pyramid Pooling'),
        (colors['C2PSA'], 'C2PSA', 'Position-aware Attention'),
        (colors['Concat'], 'Concat', 'Concatenation'),
        (colors['Segment'], 'Segment', 'Detection + Mask Head'),
    ]
    
    for i, (color, label, description) in enumerate(legend_items):
        y = legend_y - i * 0.35
        # Color box
        box = FancyBboxPatch((legend_x - 0.3, y - 0.12), 0.25, 0.25,
                             boxstyle="round,pad=0.02",
                             edgecolor='black', facecolor=color, linewidth=1)
        ax.add_patch(box)
        # Text
        ax.text(legend_x + 0.05, y, f'{label}:', fontsize=9, fontweight='bold', va='center')
        ax.text(legend_x + 0.8, y, description, fontsize=8, va='center', style='italic')
    
    # Arrow legend
    arrow_legend_y = legend_y - len(legend_items) * 0.35 - 0.5
    ax.text(legend_x, arrow_legend_y, 'Connections:', fontsize=10, fontweight='bold')
    
    # Solid arrow
    ax.plot([legend_x - 0.2, legend_x + 0.2], [arrow_legend_y - 0.3, arrow_legend_y - 0.3], 
            'k-', linewidth=2.5)
    ax.arrow(legend_x + 0.15, arrow_legend_y - 0.3, 0.05, 0, 
             head_width=0.08, head_length=0.05, fc='black', ec='black')
    ax.text(legend_x + 0.5, arrow_legend_y - 0.3, 'Sequential flow', fontsize=8, va='center')
    
    # Dashed blue arrow
    ax.plot([legend_x - 0.2, legend_x + 0.2], [arrow_legend_y - 0.6, arrow_legend_y - 0.6], 
            'b--', linewidth=2, alpha=0.7)
    ax.arrow(legend_x + 0.15, arrow_legend_y - 0.6, 0.05, 0,
             head_width=0.08, head_length=0.05, fc='blue', ec='blue', alpha=0.7)
    ax.text(legend_x + 0.5, arrow_legend_y - 0.6, 'Skip connection (upward)', fontsize=8, va='center')
    
    # Dashed green arrow
    ax.plot([legend_x - 0.2, legend_x + 0.2], [arrow_legend_y - 0.9, arrow_legend_y - 0.9],
            'g--', linewidth=2, alpha=0.7)
    ax.arrow(legend_x + 0.15, arrow_legend_y - 0.9, 0.05, 0,
             head_width=0.08, head_length=0.05, fc='green', ec='green', alpha=0.7)
    ax.text(legend_x + 0.5, arrow_legend_y - 0.9, 'Skip connection (downward)', fontsize=8, va='center')
    
    # Box content explanation
    info_x = 20
    info_y = 2.5
    ax.text(info_x, info_y + 0.3, 'Box Information:', fontsize=11, fontweight='bold')
    ax.text(info_x, info_y, '① Layer index (circle)', fontsize=8)
    ax.text(info_x, info_y - 0.25, '② Module type (bold)', fontsize=8)
    ax.text(info_x, info_y - 0.5, '③ Channel dimensions', fontsize=8, style='italic')
    ax.text(info_x, info_y - 0.75, '④ Parameter count', fontsize=8, color='darkred', fontweight='bold')
    ax.text(info_x, info_y - 1.0, '⑤ Feature map size', fontsize=8, color='darkblue')
    
    plt.tight_layout()
    plt.savefig('yolo_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    print('Architecture diagram saved as: yolo_architecture.png')
    plt.show()


def plot_parameter_distribution():
    """Create a bar chart of parameter distribution across layers"""
    
    layer_params = {
        '0-Conv': 928,
        '1-Conv': 18560,
        '2-C3k2': 23040,
        '3-Conv': 147712,
        '4-C3k2': 91136,
        '5-Conv': 590336,
        '6-C3k2': 296448,
        '7-Conv': 1180672,
        '8-C3k2': 1182720,
        '9-SPPF': 656896,
        '10-C2PSA': 1314432,
        '13-C3k2': 427520,
        '16-C3k2': 123648,
        '17-Conv': 147712,
        '19-C3k2': 329216,
        '20-Conv': 590336,
        '22-C3k2': 1313792,
        '23-Segment': 423449,
    }
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    layers = list(layer_params.keys())
    params = list(layer_params.values())
    colors_list = ['#FF6B6B' if 'Conv' in l else '#4ECDC4' if 'C3k2' in l 
                   else '#95E1D3' if 'SPPF' in l else '#F38181' if 'C2PSA' in l
                   else '#BC6C25' for l in layers]
    
    ax1.barh(layers, params, color=colors_list, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax1.set_title('Parameter Distribution by Layer', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Format x-axis
    ax1.set_xscale('log')
    ax1.set_xlim(500, max(params) * 1.2)
    
    # Pie chart by module type
    module_params = {}
    for layer, param in layer_params.items():
        module_type = layer.split('-')[1]
        module_params[module_type] = module_params.get(module_type, 0) + param
    
    colors_pie = ['#FF6B6B', '#4ECDC4', '#95E1D3', '#F38181', '#BC6C25']
    explode = [0.05] * len(module_params)
    
    wedges, texts, autotexts = ax2.pie(module_params.values(), 
                                        labels=module_params.keys(),
                                        autopct='%1.1f%%',
                                        colors=colors_pie[:len(module_params)],
                                        explode=explode,
                                        shadow=True,
                                        startangle=90)
    
    for text in texts:
        text.set_fontsize(11)
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax2.set_title('Parameter Distribution by Module Type', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('parameter_distribution.png', dpi=300, bbox_inches='tight')
    print('Parameter distribution saved as: parameter_distribution.png')
    plt.show()


def plot_feature_map_progression():
    """Visualize how feature map sizes change through the network"""
    
    # Layer index, output channels, spatial size
    progression = [
        (0, 32, 320),
        (1, 64, 160),
        (2, 128, 160),
        (3, 128, 80),
        (4, 256, 80),
        (5, 256, 40),
        (6, 256, 40),
        (7, 512, 20),
        (8, 512, 20),
        (9, 512, 20),
        (10, 512, 20),
        (11, 512, 40),  # Upsample
        (13, 256, 40),
        (14, 256, 80),  # Upsample
        (16, 128, 80),
        (17, 128, 40),
        (19, 256, 40),
        (20, 256, 20),
        (22, 512, 20),
    ]
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    layers = [p[0] for p in progression]
    channels = [p[1] for p in progression]
    spatial = [p[2] for p in progression]
    
    # Plot channels
    ax1 = ax
    color1 = '#FF6B6B'
    ax1.set_xlabel('Layer Index', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Output Channels', fontsize=12, fontweight='bold', color=color1)
    line1 = ax1.plot(layers, channels, color=color1, marker='o', linewidth=2.5, 
                     markersize=8, label='Output Channels')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Plot spatial size
    ax2 = ax1.twinx()
    color2 = '#4ECDC4'
    ax2.set_ylabel('Spatial Resolution (H=W)', fontsize=12, fontweight='bold', color=color2)
    line2 = ax2.plot(layers, spatial, color=color2, marker='s', linewidth=2.5,
                     markersize=8, label='Spatial Size', linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Title
    ax.set_title('Feature Map Progression Through Network', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)
    
    # Annotate key points
    ax1.annotate('Backbone', xy=(5, 300), xytext=(5, 450),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=12, fontweight='bold')
    ax1.annotate('Neck + Head', xy=(16, 200), xytext=(16, 350),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('feature_map_progression.png', dpi=300, bbox_inches='tight')
    print('Feature map progression saved as: feature_map_progression.png')
    plt.show()


def print_model_summary():
    """Print detailed model summary"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOCustom(num_classes=15).to(device)
    
    print('\n' + '='*80)
    print(' '*25 + 'YOLO CUSTOM MODEL SUMMARY')
    print('='*80)
    
    total_params = count_parameters(model)
    print(f'\nTotal Parameters: {total_params:,}')
    print(f'Device: {device}')
    
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    # Test with dummy input
    x = torch.randn(1, 3, 640, 640).to(device)
    
    with torch.no_grad():
        output = model(x)
    
    print('\n' + '-'*80)
    print('OUTPUT SHAPES:')
    print('-'*80)
    print(f"  Detections P3 (80x80):   {output['detections'][0].shape}")
    print(f"  Detections P4 (40x40):   {output['detections'][1].shape}")
    print(f"  Detections P5 (20x20):   {output['detections'][2].shape}")
    print(f"  Masks P3 (80x80):        {output['masks'][0].shape}")
    print(f"  Masks P4 (40x40):        {output['masks'][1].shape}")
    print(f"  Masks P5 (20x20):        {output['masks'][2].shape}")
    print(f"  Prototypes (320x320):    {output['protos'].shape}")
    
    print('\n' + '-'*80)
    print('LAYER-WISE PARAMETER COUNT:')
    print('-'*80)
    
    backbone_params = 0
    neck_params = 0
    head_params = sum(p.numel() for p in model.seg.parameters())
    
    for i, layer in enumerate(model.layers):
        params = sum(p.numel() for p in layer.parameters())
        if params > 0:
            print(f"  Layer {i:2d} ({layer.__class__.__name__:8s}): {params:>12,} params")
            if i <= 10:
                backbone_params += params
            else:
                neck_params += params
    
    print(f"  Segment Head:            {head_params:>12,} params")
    
    print('\n' + '-'*80)
    print('PARAMETER BREAKDOWN:')
    print('-'*80)
    print(f"  Backbone:     {backbone_params:>12,} params ({backbone_params/total_params*100:.1f}%)")
    print(f"  Neck:         {neck_params:>12,} params ({neck_params/total_params*100:.1f}%)")
    print(f"  Head:         {head_params:>12,} params ({head_params/total_params*100:.1f}%)")
    print(f"  {'─'*76}")
    print(f"  TOTAL:        {total_params:>12,} params")
    print('='*80 + '\n')


if __name__ == '__main__':
    print("Generating YOLO Model Visualizations...\n")
    
    # Print model summary
    print_model_summary()
    
    # Generate visualizations
    print("\nGenerating architecture diagram...")
    visualize_architecture()
    
    print("\nGenerating parameter distribution charts...")
    plot_parameter_distribution()
    
    print("\nGenerating feature map progression chart...")
    plot_feature_map_progression()
    
    print("\n✅ All visualizations completed!")
    print("\nGenerated files:")
    print("  - yolo_architecture.png")
    print("  - parameter_distribution.png")
    print("  - feature_map_progression.png")
