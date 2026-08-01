from flask import Flask, render_template, jsonify, request, send_file
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # MUST BE SET BEFORE IMPORTING PYPLOT
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

# Constants
SPATIAL_DIR = r"data/processed/generator/spatial_maps"
META_CSV = r"data/processed/classification/tcga_subtypes/manifests/legacy_10k_samples.csv"
LABELS_CSV = r"data/processed/classification/tcga_subtypes/manifests/legacy_curated_samples.csv"

# Load Meta
meta_df = pd.read_csv(META_CSV)

# Load or init labels
if not os.path.exists(LABELS_CSV):
    pd.DataFrame(columns=["image_path", "image_name", "label", "is_good"]).to_csv(LABELS_CSV, index=False)
    
labels_df = pd.read_csv(LABELS_CSV)
annotated_names = set(labels_df['image_name'].tolist())

# Create an index of remaining images
remaining_df = meta_df[~meta_df['image_path'].apply(lambda x: os.path.basename(x).replace('.png', '')).isin(annotated_names)]
remaining_df = remaining_df.sample(frac=1).reset_index(drop=True) # Shuffle
current_idx = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/next_image')
def next_image():
    global current_idx, remaining_df
    if current_idx >= len(remaining_df):
        return jsonify({"error": "No more images to annotate!"})
        
    row = remaining_df.iloc[current_idx]
    img_path = row['image_path']
    img_name = os.path.basename(img_path).replace('.png', '')
    
    return jsonify({
        "image_path": img_path,
        "image_name": img_name,
        "label": row['label'],
        "remaining": len(remaining_df) - current_idx
    })

@app.route('/api/submit', methods=['POST'])
def submit():
    global current_idx
    data = request.json
    
    new_row = pd.DataFrame([{
        "image_path": data['image_path'],
        "image_name": data['image_name'],
        "label": data['label'],
        "is_good": data['is_good']
    }])
    
    new_row.to_csv(LABELS_CSV, mode='a', header=False, index=False)
    current_idx += 1
    return jsonify({"success": True})

@app.route('/image/raw/<image_name>')
def get_raw_image(image_name):
    row = meta_df[meta_df['image_path'].str.contains(image_name)].iloc[0]
    return send_file(row['image_path'], mimetype='image/png')

@app.route('/image/spatial/<image_name>')
def get_spatial_map(image_name):
    spatial_path = os.path.join(SPATIAL_DIR, f"{image_name}.npz")
    try:
        # Load map. Shape is (512, 512, 5)
        spatial_map = np.load(spatial_path)['map']
        spatial_map = spatial_map.astype(np.float32) / 255.0
        
        # Plot 5 individual channels
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        titles = ['Tumor', 'Immune', 'Stroma', 'Necrosis', 'Other']
        
        for i in range(5):
            if i < spatial_map.shape[2]:
                axes[i].imshow(spatial_map[:, :, i], cmap='magma')
            axes[i].set_title(titles[i], color='white', fontsize=18)
            axes[i].axis('off')
            
        fig.patch.set_facecolor('#1e1e1e')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='#1e1e1e')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')
    except Exception as e:
        print("Error loading spatial map:", str(e))
        # Return empty black image on error
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(np.zeros((512, 512)))
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='black')
        buf.seek(0)
        plt.close(fig)
        return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    # Make templates dir
    os.makedirs('templates', exist_ok=True)
    app.run(debug=True, port=5000)
