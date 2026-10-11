# Netflix Data Analysis and Machine Learning

## Project Overview
This project explores the Netflix titles dataset using machine learning techniques. It implements four tasks: a content recommendation system, content type classification, audience rating classification, and content segmentation.

## Dataset
- Source file: `DATASET.csv`
- Records: 8,790 titles
- Features include title, type, director, country, rating, release year, duration, and genre categories.

## Tasks Implemented

### Task 1: Recommendation System
Used TF-IDF vectorization and cosine similarity on available metadata to recommend titles with similar metadata.

### Task 2: Content Type Classification
Trained Decision Tree and Random Forest classifiers to predict whether a title is a Movie or TV Show using metadata features.

### Task 3: Audience Rating Classification
Trained Decision Tree and Random Forest models to predict content rating categories. The Decision Tree achieved 41.01% test accuracy, while the Random Forest achieved 34.58%.

### Task 4: Content Segmentation
Applied K-Means clustering with six clusters using encoded metadata and release year. Saved title-level cluster assignments, cluster summaries, and a two-dimensional visualization.

## Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- TF-IDF, cosine similarity, Decision Tree, Random Forest, K-Means

## How to Run
1. Install Python 3.11 or a compatible Python version.
2. Install the required libraries: `pandas`, `numpy`, `scikit-learn`, and `matplotlib`.
3. Place `DATASET.csv` in the project root directory.
4. Run the scripts from the project folder:

   - `python task1_recommendation.py`
   - `python task2_content_type.py`
   - `python task3_rating_classification.py`
   - `python task4_content_segmentation.py`

## Output
The `outputs/` folder contains generated recommendations, model comparisons, confusion matrices, cluster assignments, cluster summaries, and visualizations.

## Limitations
The recommendation system uses the available metadata rather than plot descriptions. Classification results depend on the available features and class distributions. Clusters represent groups based on encoded metadata and should not be interpreted as official Netflix categories.