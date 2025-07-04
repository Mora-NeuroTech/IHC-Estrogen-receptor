# IHC-Estrogen-receptor

# 🧪 Histological Image Cell Analysis

This Python script performs segmentation and statistical analysis of blue and brown-stained cells in histology slide images. It uses watershed segmentation, morphological processing, and grayscale intensity analysis to classify and quantify cells.

## 🔬 Key Features

* ✅ Automatically detects **blue** and **brown** cells in RGB images.
* 📏 Computes **area percentages** and **cell counts** for both masks.
* 📊 Analyzes **grayscale intensity** of brown-stained regions.
* 🧠 Generates a **status score** based on stain coverage and intensity.
* 📸 Visualizes segmentation results and intensity distributions.

---

## 🖼️ Example Input

You must provide a histology slide image (`.png`, `.jpg`, etc.) as input.

```bash
slide (1).png  # Replace with your image file
```
---

## 📂 File Structure

* `main.py` – Core script for loading image, processing, and visualizing results.
* `README.md` – Documentation for repository (this file).

---

## ⚙️ Dependencies

Install all required libraries using pip:

```bash
pip install opencv-python scikit-image matplotlib numpy scipy
```

---

## ▶️ How to Run

1. **Add your image** to the directory (e.g., `slide (1).png`)
2. **Run the script**:

```bash
python main.py
```

3. **Outputs will include**:

   * Detected cell counts and percentages.
   * Grayscale intensity statistics.
   * Final classification: `Negative`, `Low Positive`, or `Positive`.
   * Plots for intensity distributions.
   * Image visualizations with segmented contours.

---

## 🧮 Methods Used

### 🔹 Blue Mask Extraction

* RGB thresholding
* Morphological cleaning
* Removal of elongated artifacts

### 🔸 Brown Mask Extraction

* LAB color space thresholding
* Distance transform & watershed
* Removal of small regions

### 📈 Intensity Analysis

* Median-based scoring:

  * Low < 0.3 → Status 3
  * Medium 0.3–0.6 → Status 2
  * High > 0.6 → Status 1

### 📊 Final Classification

Combines:

* **Brown area status** (based on area %)
* **Intensity status** (based on grayscale intensity)
---

## 🛠️ Functions Overview

| Function                          | Purpose                              |
| --------------------------------- | ------------------------------------ |
| `load_image()`                    | Loads and normalizes the input image |
| `create_color_masks_rgb()`        | Creates blue and brown masks         |
| `calculate_mask_stats()`          | Computes area %, cell counts         |
| `calculate_grayscale_intensity()` | Analyzes grayscale stats             |
| `calculate_marks()`               | Generates final classification       |
| `visualize_stats()`               | Shows original + labeled regions     |
| `visualize_results()`             | Displays contours on original        |
| `plot_intensity_distribution()`   | Histogram of intensity values        |

---

