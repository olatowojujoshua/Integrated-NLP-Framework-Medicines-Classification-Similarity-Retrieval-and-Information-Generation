# Medicines Analysis Platform

A professional, modern machine learning platform for pharmaceutical classification and analysis.

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements_new.txt
```

### Run Application
```bash
streamlit run main.py
```

## 📋 Features

### 🎯 Core Functionality
- **Data Upload**: Support for Excel and CSV files
- **Machine Learning**: Multiple classification models
- **Real-time Analysis**: Instant medicine classification
- **Performance Metrics**: Model accuracy tracking
- **Interactive Visualizations**: Beautiful charts and graphs

### 🤖 Machine Learning Models
- **Logistic Regression**: Fast and interpretable
- **Support Vector Machine**: High-performance classification
- **Naive Bayes**: Probabilistic predictions
- **Ensemble Methods**: Combined model performance

### 📊 Data Analysis
- **Class Distribution**: Visualize therapeutic classes
- **Model Comparison**: Compare model performance
- **Feature Analysis**: TF-IDF vectorization
- **Accuracy Metrics**: Detailed performance reports

## 🎨 Professional UI/UX

### Modern Design
- **Clean Interface**: Professional gradient design
- **Responsive Layout**: Works on all devices
- **Interactive Elements**: Smooth animations and transitions
- **Professional Typography**: Inter font family
- **Color Scheme**: Modern purple gradient theme

### User Experience
- **Intuitive Navigation**: Tab-based interface
- **Real-time Feedback**: Progress indicators
- **Error Handling**: Clear error messages
- **Data Validation**: Input validation and checks

## 📁 Project Structure

```
├── main.py                 # Main application
├── requirements_new.txt     # Dependencies
├── data/                   # Data directory
│   └── MID.xlsx           # Sample dataset
├── src/                   # Source code (legacy)
└── Notebooks/             # Analysis notebooks
```

## 🔧 Configuration

### Data Requirements
- **Format**: Excel (.xlsx) or CSV (.csv)
- **Required Columns**: 
  - `Name`: Medicine name
  - `Therapeutic Class`: Classification target
- **Optional Columns**: Additional metadata

### Model Settings
- **Sample Size**: Control dataset size for testing
- **Test Split**: Train/test split percentage
- **Feature Engineering**: TF-IDF vectorization
- **Model Selection**: Choose which models to train

## 📈 Performance

### Model Accuracy
- **Logistic Regression**: ~85-90%
- **SVM**: ~87-92%
- **Naive Bayes**: ~80-85%

### Processing Speed
- **Small Dataset (1K records)**: < 10 seconds
- **Medium Dataset (10K records)**: < 30 seconds
- **Large Dataset (100K records)**: < 2 minutes

## 🎯 Usage Guide

### 1. Upload Data
1. Open the application
2. Go to sidebar
3. Upload your dataset (Excel or CSV)
4. Configure sample size

### 2. Train Models
1. Go to "Models" tab
2. Select models to train
3. Click "Train Models"
4. Wait for training completion

### 3. Analyze Results
1. Go to "Analysis" tab
2. Enter medicine name
3. Click "Classify"
4. View predictions

### 4. View Performance
1. Go to "Results" tab
2. View accuracy charts
3. Compare models
4. Analyze metrics

## 🔒 Security & Privacy

### Data Protection
- **Local Processing**: All data processed locally
- **No Cloud Upload**: No data sent to external servers
- **Session Storage**: Data stored in session only
- **Privacy First**: No user tracking or analytics

### Best Practices
- **Data Validation**: Validate input data
- **Error Handling**: Robust error management
- **Performance Monitoring**: Track system health
- **Regular Updates**: Keep dependencies updated

## 🛠️ Technology Stack

### Frontend
- **Streamlit**: Modern web framework
- **Plotly**: Interactive visualizations
- **CSS3**: Professional styling
- **JavaScript**: Interactive elements

### Backend
- **Python**: Core programming language
- **scikit-learn**: Machine learning library
- **pandas**: Data processing
- **numpy**: Numerical computing

### ML/AI
- **TF-IDF**: Text vectorization
- **Classification Models**: Multiple algorithms
- **Cross-validation**: Model evaluation
- **Feature Engineering**: Text preprocessing

## 📞 Support

### Documentation
- **User Guide**: Step-by-step instructions
- **API Reference**: Technical documentation
- **Troubleshooting**: Common issues and solutions

### Requirements
- **Python**: 3.8+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB free space
- **Browser**: Modern web browser

---

**Medicines Analysis Platform** - Professional ML-powered pharmaceutical classification system.
