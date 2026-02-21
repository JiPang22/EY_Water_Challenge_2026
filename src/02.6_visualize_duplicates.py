import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.image as mpimg

# Define constants from the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'rawData')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
IMAGE_CHIPS_DIR = os.path.join(RAW_DATA_DIR, 'chips')

# Ensure the plots directory exists
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load the training data metadata
training_labels_df = pd.read_csv(os.path.join(RAW_DATA_DIR, 'water_quality_training_dataset.csv'))
training_labels_df.columns = training_labels_df.columns.str.strip()

# Convert 'Sample Date' column to datetime objects
training_labels_df['dt'] = pd.to_datetime(training_labels_df['Sample Date'], dayfirst=True, errors='coerce')

# Find duplicates by date
duplicate_dates = training_labels_df[training_labels_df.duplicated(subset=['dt'], keep=False)]

print("Some duplicate dates found:")
print(duplicate_dates['dt'].value_counts().head())


# Visualize all images for a specific date: 2014-01-08
target_date = datetime(2014, 1, 8)
specific_date_df = training_labels_df[training_labels_df['dt'] == target_date]

print(f"\nFound {len(specific_date_df)} entries for the date {target_date.date()}.")

if not specific_date_df.empty:
    # Determine grid size
    num_images = len(specific_date_df)
    if num_images > 0:
        cols = int(np.ceil(np.sqrt(num_images)))
        rows = int(np.ceil(num_images / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.5))
        if num_images > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        for i, (idx, row) in enumerate(specific_date_df.iterrows()):
            ax = axes[i]
            
            try:
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
                loc_id = f"{lat:.4f}_{lon:.4f}"
                date_str = row['dt'].strftime("%Y-%m-%d")
                img_path = os.path.join(IMAGE_CHIPS_DIR, f"loc_{loc_id}", f"{date_str}.npy")

                if os.path.exists(img_path):
                    # Load the image chip (Shape: 5, 32, 32 -> R, G, B, NIR, SWIR)
                    img_chip = np.load(img_path).astype(np.float32)

                    # Based on patch_generator_v6.py: TARGET_BANDS = ["B02", "B03", "B04", "B08"]
                    # B02=Blue, B03=Green, B04=Red. So indices are [0, 1, 2] -> [Blue, Green, Red]
                    # For RGB visualization, we need [Red, Green, Blue] -> indices [2, 1, 0]
                    rgb_img = img_chip[[2, 1, 0], :, :]

                    # Transpose to (H, W, C) for imshow
                    rgb_img = np.transpose(rgb_img, (1, 2, 0))

                    # Robust Normalization (2% - 98% percentile stretch)
                    p2, p98 = np.percentile(rgb_img, (2, 98))
                    if p98 > p2:
                        rgb_img = (rgb_img - p2) / (p98 - p2)
                    else:
                        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-6)
                    
                    rgb_img = np.clip(rgb_img, 0, 1)

                    ax.imshow(rgb_img)
                    ax.set_title(f"Lat: {lat:.4f}\nLon: {lon:.4f}", fontsize=8)
                else:
                    ax.text(0.5, 0.5, "Not Found", ha='center', va='center')
                    ax.set_title(f"Lat: {lat:.4f}\nLon: {lon:.4f}", fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, "Error", ha='center', va='center')
                print(f"Error processing row {idx}: {e}")

            ax.axis('off')

        # Hide unused subplots
        for i in range(num_images, len(axes)):
            axes[i].axis('off')

        plt.suptitle(f'Image Chips for {target_date.strftime("%Y-%m-%d")}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(PLOTS_DIR, '02.6_duplicate_date_2014-01-08_images.png')
        plt.savefig(save_path)
        print(f"\nSaved image grid to: {save_path}")
        plt.close()
    else:
        print("No images found for the specified date.")

else:
    print(f"No data found for the date {target_date.date()}.")
