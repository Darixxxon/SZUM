import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from scipy import stats
import argparse

PLOT_DIR = Path("Plots")
DATASET_ROOT = Path("dataset_augmented")
IMG_SIZE = (256, 256)  # Resize for consistency, as in dataset_preparation.py

def load_images_and_colors(root: Path):
    """
    Load all images from dataset and compute mean RGB/intensity values per image.
    Handles both RGB and grayscale images.
    Returns: dict of class -> list of (r, g, b) tuples
    """
    color_data = {}
    total_images = 0

    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        color_data[class_name] = []

        for img_path in class_dir.glob("*.jpg"):
            try:
                img = Image.open(img_path).resize(IMG_SIZE)
                img_array = np.array(img)

                if img.mode == 'L':  # Grayscale
                    mean_intensity = np.mean(img_array)
                    mean_r = mean_g = mean_b = mean_intensity
                else:  # RGB or other
                    mean_r = np.mean(img_array[:, :, 0])
                    mean_g = np.mean(img_array[:, :, 1])
                    mean_b = np.mean(img_array[:, :, 2])

                color_data[class_name].append((mean_r, mean_g, mean_b))
                total_images += 1
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue

    print(f"Loaded {total_images} images from {len(color_data)} classes.")
    return color_data

def perform_statistical_tests(color_data):
    """
    Perform ANOVA on R, G, B channels across classes.
    """
    classes = list(color_data.keys())
    r_values = [np.array([c[0] for c in color_data[cls]]) for cls in classes]
    g_values = [np.array([c[1] for c in color_data[cls]]) for cls in classes]
    b_values = [np.array([c[2] for c in color_data[cls]]) for cls in classes]

    # ANOVA for each channel
    r_f, r_p = stats.f_oneway(*r_values)
    g_f, g_p = stats.f_oneway(*g_values)
    b_f, b_p = stats.f_oneway(*b_values)

    print("ANOVA Results:")
    print(f"Red channel: F={r_f:.2f}, p={r_p:.4e} {'(significant)' if r_p < 0.05 else '(not significant)'}")
    print(f"Green channel: F={g_f:.2f}, p={g_p:.4e} {'(significant)' if g_p < 0.05 else '(not significant)'}")
    print(f"Blue channel: F={b_f:.2f}, p={b_p:.4e} {'(significant)' if b_p < 0.05 else '(not significant)'}")

    return r_p, g_p, b_p

def visualize_color_distributions(color_data, dataset_name):
    """
    Create boxplots for RGB means per class (first 10 classes for readability).
    """
    classes = list(color_data.keys())[:10]  # Limit to first 10 for readability
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Red
    axes[0].boxplot([ [c[0] for c in color_data[cls]] for cls in classes ], labels=classes, vert=False)
    axes[0].set_title('Red Channel Means per Class (First 10)')
    axes[0].set_xlabel('Mean Red Value')

    # Green
    axes[1].boxplot([ [c[1] for c in color_data[cls]] for cls in classes ], labels=classes, vert=False)
    axes[1].set_title('Green Channel Means per Class (First 10)')
    axes[1].set_xlabel('Mean Green Value')

    # Blue
    axes[2].boxplot([ [c[2] for c in color_data[cls]] for cls in classes ], labels=classes, vert=False)
    axes[2].set_title('Blue Channel Means per Class (First 10)')
    axes[2].set_xlabel('Mean Blue Value')

    plt.tight_layout()
    PLOT_DIR.mkdir(exist_ok=True)
    safe_name = dataset_name.lower().replace(' ', '_')
    plot_filename = f"{safe_name}_color_bias_boxplots.png"
    plt.savefig(PLOT_DIR / plot_filename)
    print(f"Saved plot to {PLOT_DIR / plot_filename}")

def identify_problematic_classes(color_data, threshold=20):
    """
    Identify classes with color variance above threshold.
    Threshold: percentage of overall variance.
    """
    all_r = [c[0] for cls in color_data for c in color_data[cls]]
    all_g = [c[1] for cls in color_data for c in color_data[cls]]
    all_b = [c[2] for cls in color_data for c in color_data[cls]]

    overall_var_r = np.var(all_r)
    overall_var_g = np.var(all_g)
    overall_var_b = np.var(all_b)

    print(f"Overall variance - R: {overall_var_r:.2f}, G: {overall_var_g:.2f}, B: {overall_var_b:.2f}")

    problematic = []
    for cls in color_data:
        cls_r = [c[0] for c in color_data[cls]]
        cls_g = [c[1] for c in color_data[cls]]
        cls_b = [c[2] for c in color_data[cls]]

        cls_var_r = np.var(cls_r)
        cls_var_g = np.var(cls_g)
        cls_var_b = np.var(cls_b)

        if cls_var_r > (threshold/100) * overall_var_r or cls_var_g > (threshold/100) * overall_var_g or cls_var_b > (threshold/100) * overall_var_b:
            problematic.append(cls)

    print(f"Problematic classes (variance > {threshold}% of overall): {problematic}")
    return problematic

def main():
    parser = argparse.ArgumentParser(description="Validate color/intensity-class correlations in dataset.")
    parser.add_argument("dataset_dir", type=str, help="Path to the dataset directory")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_dir)
    if not dataset_root.exists():
        print(f"Dataset directory {dataset_root} not found.")
        return

    print(f"Processing dataset: {dataset_root}")
    print("Loading images and computing color/intensity statistics...")
    color_data = load_images_and_colors(dataset_root)

    print("Performing statistical tests...")
    r_p, g_p, b_p = perform_statistical_tests(color_data)

    print("Generating visualizations...")
    visualize_color_distributions(color_data, dataset_root.name)

    print("Identifying problematic classes...")
    problematic = identify_problematic_classes(color_data)
    print(f"Number of problematic classes: {len(problematic)}/{len(color_data)}")

    print("\nSummary:")
    if any(p < 0.05 for p in [r_p, g_p, b_p]):
        print("Significant color/intensity-class correlations detected. Model may learn biases.")
        print("Problematic classes:", problematic)
    else:
        print("No significant color/intensity-class correlations found. Proceed with caution.")

if __name__ == "__main__":
    main()