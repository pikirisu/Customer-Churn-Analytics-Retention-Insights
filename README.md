# Customer Analytics with Artificial Neural Networks

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning project implementing **Artificial Neural Networks (ANNs)** for customer analytics, featuring both **binary classification** (churn prediction) and **regression** (salary estimation) tasks. This project demonstrates end-to-end machine learning engineering from data preprocessing to production-ready web applications.

##  Project Overview

This project solves real-world business problems using deep learning:

1. **Customer Churn Prediction** (Classification): Predicts the probability of customer churn using customer demographics and account features
2. **Salary Estimation** (Regression): Estimates customer salary to identify high-value customers for targeted marketing strategies

### Live Demo
 **Deployed Application**: [Customer Churn Prediction App](https://ann-classification-churn-apa3zvgpebjoxfp8jhwaza.streamlit.app/)

## ✨ Key Features

- **Dual Model Architecture**: Separate ANN models for classification and regression tasks
- **Production-Ready Web Interface**: Interactive Streamlit applications with real-time predictions
- **Comprehensive Data Preprocessing**: Feature engineering, encoding, and scaling pipelines
- **Model Persistence**: Serialized models and preprocessors for seamless deployment
- **Hyperparameter Tuning**: Systematic approach to model optimization
- **Training Visualization**: TensorBoard integration for monitoring training metrics
- **Early Stopping**: Prevents overfitting with intelligent callback mechanisms

##  Technical Stack

### Core Technologies
- **Deep Learning**: TensorFlow/Keras for ANN implementation
- **Data Processing**: Pandas, NumPy for data manipulation
- **Machine Learning**: scikit-learn for preprocessing and evaluation
- **Web Framework**: Streamlit for interactive applications
- **Visualization**: TensorBoard for training metrics, Matplotlib for analysis
- **Model Tuning**: scikeras for Keras-sklearn integration

### Skills Demonstrated

####  Machine Learning & Deep Learning
- **Neural Network Architecture Design**: Multi-layer perceptron (MLP) with optimized hidden layers
- **Binary Classification**: Sigmoid activation with binary cross-entropy loss
- **Regression Analysis**: Linear output layer with appropriate loss functions
- **Hyperparameter Optimization**: Learning rate tuning, architecture search
- **Model Evaluation**: Accuracy, loss metrics, validation strategies

####  Data Engineering
- **Feature Engineering**: Categorical encoding (One-Hot, Label Encoding)
- **Data Scaling**: StandardScaler for feature normalization
- **Data Validation**: Input range validation and preprocessing pipelines
- **Model Artifacts Management**: Pickle serialization for encoders and scalers

####  Software Engineering
- **Production Deployment**: Streamlit cloud deployment
- **Code Organization**: Modular structure with separation of concerns
- **Model Versioning**: Saved model formats (.h5) for reproducibility
- **Error Handling**: Input validation and user-friendly error messages

#### 📈 MLOps & Experimentation
- **Experiment Tracking**: TensorBoard logging for training visualization
- **Model Monitoring**: Early stopping callbacks to prevent overfitting
- **Reproducibility**: Fixed random seeds, consistent preprocessing pipelines
- **Model Persistence**: Complete pipeline serialization (models + preprocessors)

## 📁 Project Structure

```
ANN_Project/
│
├── app.py                      # Customer Churn Prediction Streamlit App
├── regression.py               # Salary Estimation Streamlit App
├── testing.py                  # Model testing utilities
├── requirements.txt            # Python dependencies
│
├── data/
│   └── Churn_Modelling.csv     # Customer dataset (~10,000 records)
│
├── models/                     # Trained models and preprocessors
│   ├── model.h5                # Churn prediction ANN model
│   ├── regression_model.h5     # Salary regression ANN model
│   ├── scaler.pkl              # Feature scaler (classification)
│   ├── scaler_reg.pkl          # Feature scaler (regression)
│   ├── one_hot_encoder.pkl     # Geography encoder (classification)
│   ├── one_hot_encoder_reg.pkl # Geography encoder (regression)
│   ├── label_encoder_gender.pkl
│   └── label_encoder_gender_reg.pkl
│
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── experiment.ipynb        # Churn model development & training
│   ├── salaryregression.ipynb  # Regression model development
│   ├── tuning_ann.ipynb        # Hyperparameter tuning strategies
│   └── prediction.ipynb        # Prediction examples
│
└── logs/                      # TensorBoard training logs
    └── fit/                    # Training run logs with timestamps
```

##  Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd ANN_Project
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"
```

## 📖 Usage

### Running the Churn Prediction App
```bash
streamlit run app.py
```
The application will open in your default browser at `http://localhost:8501`

### Running the Salary Estimation App
```bash
streamlit run regression.py
```

### Using the Models Programmatically

#### Churn Prediction
```python
import tensorflow as tf
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model and preprocessors
model = tf.keras.models.load_model('models/model.h5')
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
# ... load encoders similarly

# Prepare input data
input_data = {
    'CreditScore': [650],
    'Geography': ['France'],
    'Gender': ['Male'],
    'Age': [45],
    'Tenure': [5],
    'Balance': [10000],
    'NumOfProducts': [2],
    'HasCrCard': [1],
    'IsActiveMember': [1],
    'EstimatedSalary': [50000]
}

# Preprocess and predict
# ... (apply encoding and scaling)
prediction = model.predict(scaled_input)
churn_probability = prediction[0][0]
```

##  Model Architecture

### Churn Prediction Model (Classification)
```
Input Layer:  12 features
    ↓
Hidden Layer 1: 64 neurons, ReLU activation
    ↓
Hidden Layer 2: 32 neurons, ReLU activation
    ↓
Output Layer: 1 neuron, Sigmoid activation
```

**Training Configuration:**
- Optimizer: Adam (learning_rate=0.001)
- Loss Function: Binary Crossentropy
- Metrics: Accuracy
- Epochs: 100 (with early stopping)
- Validation Split: Standard train-test split
- Callbacks: EarlyStopping, TensorBoard

### Salary Regression Model
```
Input Layer:  12 features (includes Exited status)
    ↓
Hidden Layer 1: 64 neurons, ReLU activation
    ↓
Hidden Layer 2: 32 neurons, ReLU activation
    ↓
Output Layer: 1 neuron (linear activation)
```

**Training Configuration:**
- Optimizer: Adam
- Loss Function: Mean Squared Error (MSE)
- Metrics: Mean Absolute Error (MAE)

##  Dataset Information

- **Source**: Churn_Modelling.csv
- **Size**: ~10,000 customer records
- **Features**:
  - **Demographic**: Age, Gender, Geography
  - **Financial**: Credit Score, Balance, Estimated Salary
  - **Account**: Tenure, Number of Products, Has Credit Card, Is Active Member
  - **Target (Classification)**: Exited (Churn Status)
  - **Target (Regression)**: Estimated Salary

### Data Preprocessing Pipeline
1. **Categorical Encoding**: 
   - Geography → One-Hot Encoding (3 categories)
   - Gender → Label Encoding
2. **Feature Scaling**: StandardScaler normalization
3. **Train-Test Split**: Standard stratified split for classification

##  Skills & Competencies Highlighted

### Technical Skills
-  **Deep Learning**: ANN architecture design, activation functions, loss functions
-  **Machine Learning**: Classification, regression, model evaluation
-  **Data Science**: EDA, feature engineering, preprocessing pipelines
-  **Python Programming**: Object-oriented design, modular code structure
-  **MLOps**: Model deployment, versioning, monitoring
-  **Web Development**: Streamlit framework, interactive UI design

### Soft Skills & Best Practices
-  **Problem-Solving**: End-to-end ML pipeline development
-  **Documentation**: Comprehensive README and code comments
-  **Version Control**: Git workflow and repository management
-  **Production Mindset**: Deployment-ready applications
-  **Experimentation**: Systematic hyperparameter tuning approach

## 📈 Model Performance

### Churn Prediction Model
- **Validation Accuracy**: ~86-87%
- **Training Strategy**: Early stopping prevents overfitting
- **Threshold**: 0.5 probability for churn classification

### Salary Regression Model
- **Evaluation**: Mean Absolute Error (MAE) and Mean Squared Error (MSE)
- **Business Logic**: Customers with predicted salary > $100,000 classified as high-value

*Note: Detailed metrics available in TensorBoard logs and training notebooks*

##  Future Enhancements

- [ ] Implement cross-validation for more robust evaluation
- [ ] Add automated hyperparameter tuning with GridSearchCV/RandomSearch
- [ ] Expand feature engineering (interaction terms, polynomial features)
- [ ] Implement model explainability (SHAP values, feature importance)
- [ ] Add batch prediction API endpoints
- [ ] Implement A/B testing framework for model comparison
- [ ] Add data drift detection and monitoring
- [ ] Create automated retraining pipeline

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

##  License

This project is open source and available under the [MIT License](LICENSE).

##  Author

**Your Name**
- GitHub: [pikirisu](https://github.com/pikirisu)
- LinkedIn: [Akshat Chaurasia](https://www.linkedin.com/in/akshat-chaurasia-1289252a9/)

## 🙏 Acknowledgments

- TensorFlow/Keras team for the excellent deep learning framework
- Streamlit for the intuitive web app framework
- scikit-learn for comprehensive ML utilities
- The open-source community for continuous inspiration

---

⭐ **Star this repository if you find it helpful!**

**Built with ❤️ for demonstrating production-ready deep learning capabilities**
