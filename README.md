# Plant Disease Prediction CNN Image Classifier

This project is a Convolutional Neural Network (CNN) based image classifier for plant disease detection using the PlantVillage dataset. The workflow and code are designed for direct use in Google Colab, making it easy to upload, run, and experiment with the project in a cloud environment.

## Project Features
- **Deep Learning Model**: Uses TensorFlow and Keras to build and train a CNN for multi-class plant disease classification.
- **Dataset**: Utilizes the PlantVillage dataset (downloaded via Kaggle API) with color, segmented, and grayscale images.
- **Colab Friendly**: Includes all necessary commands for running in Google Colab (e.g., pip installs, Kaggle API setup).
- **Visualization**: Plots training/validation accuracy and loss.
- **Prediction Utility**: Functions for loading, preprocessing, and predicting new images.

## How to Use in Google Colab

1. **Upload the Project**
   - Upload all project files (including the main Python script and this README) to your Google Colab environment.

2. **Install Dependencies**
   - The script includes commands to install required packages:
     ```python
     !pip install kaggle
     ```

3. **Kaggle API Credentials**
   - Upload your `kaggle.json` file (Kaggle API token) to the Colab environment.
   - The script will automatically read this file and set up the environment variables.

4. **Download and Extract Dataset**
   - The script uses the Kaggle API to download the PlantVillage dataset and extracts it for use.

5. **Run the Main Script**
   - Execute the code cells in order. The script will:
     - Explore the dataset
     - Set up data generators
     - Build and train the CNN model
     - Evaluate and visualize results
     - Save the trained model and class indices

6. **Predict New Images**
   - Use the provided utility functions to preprocess and predict the class of new leaf images.
   - Example usage is included at the end of the script.

## File Structure
```
Plant_Disease_Prediction_CNN_Image_Classifier/
│
├── Plant_Disease_Prediction_CNN_Image_Classifier.py  # Main script (Colab-ready)
├── README.md                                        # This file
├── plant disease/
│   └── plantvillage dataset/
│       ├── color/
│       ├── segmented/
│       └── grayscale/
└── class_indices.json                               # Generated after training
```

## Notes
- **Colab Paths**: The script uses `/content/` paths for files and images. Adjust if running elsewhere.
- **Kaggle Dataset**: Make sure your Kaggle account has access to the dataset.
- **Training Time**: Training may take several minutes depending on the GPU provided by Colab.

## References
- [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)

---

