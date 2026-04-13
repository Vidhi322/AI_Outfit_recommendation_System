# Fashion Recommendation System

## 📋 Overview

The Fashion Recommendation System is an intelligent outfit recommendation engine powered by deep learning. It uses a multi-input Convolutional Neural Network (CNN) to analyze clothing items and occasions, providing personalized outfit suggestions with compatibility ratings. The system combines a sophisticated machine learning model with an intuitive web application, allowing users to manage their virtual wardrobe and receive outfit recommendations tailored to different occasions.

## ✨ Key Features

- **Intelligent Outfit Recommendations**: Multi-input CNN model that evaluates topwear, bottomwear, footwear, and occasion context
- **Compatibility Scoring**: Provides ratings from 0 to 1 indicating outfit suitability for a given occasion
- **Virtual Wardrobe Management**: Create and organize your digital closet with your personal clothing collection
- **Occasion-Based Suggestions**: Get outfit recommendations tailored to specific events (casual, office, party, sports, etc.)
- **Next Best Outfit**: Discover alternative outfit combinations based on your preferences
- **User Authentication**: Secure login system for personalized wardrobe management
- **Responsive Web Interface**: Easy-to-use interface for uploading clothes and viewing recommendations

## 🛠️ Technology Stack

- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow, Keras, CNN
- **Frontend**: HTML, CSS, JavaScript
- **Database**: SQLite/File-based storage
- **Image Processing**: NumPy, image classification models
- **Deployment**: Heroku (Procfile included)

## 📁 Project Structure

```
fashion-recommendation-system/
├── app.py                          # Main Flask application
├── recommender.py                  # Core recommendation engine
├── train_model.py                  # Model training script
├── wardrobe.py                     # Wardrobe management module
├── requirements.txt                # Python dependencies
├── Procfile                        # Heroku deployment configuration
├── models/                         # Pre-trained ML models
│   ├── fashion-recommendation1.h5
│   ├── keras_model.h5
│   └── type_model.h5
├── static/                         # CSS and styling
│   └── custom.css
├── templates/                      # HTML templates
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── fashion.html
│   ├── profile.html
│   └── ...
├── users/                          # User wardrobe data
│   └── [username]/
│       ├── pants/
│       ├── shirt/
│       ├── t-shirt/
│       ├── shoes/
│       └── party/
└── fcs_dataset/                    # Training dataset
    ├── topwear/
    ├── bottomwear/
    └── footwear/
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Vidhi322/AI_Outfit_recommendation_System.git
   cd fashion-recommendation-System
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`
   - Create an account or login with existing credentials
   - Start building your virtual wardrobe

## 💻 How to Use

### 1. Sign Up / Login
- Create a new account with your username and password
- Or login with existing credentials

### 2. Build Your Wardrobe
- Navigate to the wardrobe section
- Upload images of your clothing items
- Categorize them by type (topwear, bottomwear, footwear) and color

### 3. Get Recommendations
- Select an occasion (office, casual, party, sports)
- Browse suggested outfit combinations ranked by compatibility
- View detailed ratings for each recommendation

### 4. Discover New Combinations
- Use the "Next Best Outfit" feature to explore alternative combinations
- Rate combinations to improve future recommendations

## 🧠 Model Architecture

The recommendation system uses a multi-input CNN architecture:

- **Input Layers**: Separate branches for topwear, bottomwear, footwear, and occasion embeddings
- **Processing Layers**: Convolutional layers with ReLU activation for feature extraction
- **Feature Fusion**: Concatenation of processed features from all inputs
- **Output Layer**: Dense layers regressing to a 0-1 compatibility score

## 📊 Dataset

The model is trained on the FCS (Fashion Clothing System) dataset containing:
- **Topwear**: Shirts, t-shirts, and similar upper garments
- **Bottomwear**: Pants, skirts, jeans, and similar lower garments
- **Footwear**: Shoes, sneakers, and other footwear items

Training data is organized by clothing type and color for enhanced classification accuracy.

## 🔄 Training the Model

To train a new recommendation model:

```bash
python train_model.py
```

This script will:
- Load and preprocess the clothing dataset
- Train the multi-input CNN model
- Evaluate performance on validation data
- Save the trained model for use in recommendations

## 🚀 Deployment

The application includes a `Procfile` for easy deployment to Heroku:

```bash
heroku create your-app-name
git push heroku main
```

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests with improvements
- Improve documentation


