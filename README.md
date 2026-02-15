## Project Overview

This project implements face detection, feature extraction, clustering, and template classification using OpenCV and K-Means clustering.
The system detects faces from an image, extracts HSV color features (Hue and Saturation), clusters similar faces using K-Means, and predicts the cluster of a given template face image.

## Aim

The aim of this project is to:

- Detect faces from an input image using Haar Cascade.
- Extract meaningful HSV color features from detected faces.
- Perform unsupervised clustering using K-Means.
- Classify a template image into one of the learned clusters.
- Visualize clustering results using graphs.

## Methodology

### Face Detection

- Converted image to grayscale.
- Used Haar Cascade Classifier from OpenCV.
- Applied `detectMultiScale()` to detect faces.
- Drew bounding boxes around detected faces.

### Feature Extraction

- Converted detected face regions from BGR to HSV color space.
- Extracted:
  - Mean Hue
  - Mean Saturation
- Constructed feature vectors in the form:

Feature Vector = (Mean Hue, Mean Saturation)
### K-Means Clustering

- Applied K-Means clustering algorithm to group similar faces.
- Distance metric used (Euclidean Distance):

$$
d = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
$$

- Faces were assigned to clusters based on minimum distance from centroids.

### Template Image Classification

- Extracted HSV features from template image.
- Used `kmeans.predict()` to determine the closest cluster.
- Assigned the template image to the nearest centroid.

## Results and Visualisations

### Face Detection Output
![face_detected](https://github.com/user-attachments/assets/23fa3772-84d8-43f0-a236-8097c82b6191)


### K-Means Clustering of Detected Faces (Hue vs Saturation)
<img width="1005" height="547" alt="output1" src="https://github.com/user-attachments/assets/79aa78dd-7a43-4666-ad0f-6460a851eb62" />


### K-Means Clustering of Faces Scatter plot(Hue vs Saturation)
<img width="1005" height="547" alt="plot1_lab5" src="https://github.com/user-attachments/assets/9079805e-6147-421c-b03d-0c34e7e8d60d" />


### Face Detection Output Shashi Tharoor
![shashi_tharoor](https://github.com/user-attachments/assets/3edc79c4-441b-4131-b93c-4505ab3e78b8)


### Template face cluster prediction (Hue vs Saturation)
<img width="1005" height="547" alt="output2" src="https://github.com/user-attachments/assets/43211da8-c7bc-414a-8002-5782bd0f889e" />


### K-Means Clustering of Faces Scatter plot for template image(Hue vs Saturation)
<img width="1005" height="547" alt="plot2_lab5" src="https://github.com/user-attachments/assets/54f53913-bb5c-45f7-bec0-2fc47d949bb4" />




## Key Findings

- Faces with similar color tones were grouped together.
- K-Means effectively separated clusters based on HSV values.
- Template image was successfully classified into the nearest cluster.
- HSV color space provides meaningful representation for color-based clustering.
- Model performance depends on lighting conditions.

## Limitations

- This is clustering, not face recognition.
- Only color-based features were used.
- Sensitive to illumination changes.
- Accuracy decreases in varied lighting conditions.

## Conclusion

This project demonstrates the use of:

- Computer Vision (OpenCV)
- Feature Extraction (HSV)
- Unsupervised Learning (K-Means)
- Distance-based classification

The system successfully clusters faces based on color similarity and predicts the class of a new template image.

Although basic, this approach builds a foundation for advanced face recognition and clustering systems.

## Future Improvements

- Use deep learning-based face embeddings.
- Apply PCA for dimensionality reduction.
- Use cosine similarity instead of Euclidean distance.
- Improve lighting normalization techniques.
- Increase dataset size for better clustering performance.

## Technologies Used

- Python
- OpenCV
- NumPy
- Matplotlib
- Scikit-Learn
- Jupyter Notebook
