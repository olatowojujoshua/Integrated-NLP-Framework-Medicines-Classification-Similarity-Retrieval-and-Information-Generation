import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os

# Page Configuration
st.set_page_config(
    page_title="Pharmaceutical Intelligence Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
def load_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            box-sizing: border-box;
        }
        
        /* Wide Sidebar Configuration - More Aggressive */
        .css-1d391kg,
        section[data-testid="stSidebar"] {
            width: 450px !important;
            min-width: 450px !important;
            max-width: 450px !important;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border-right: 1px solid #cbd5e0;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }
        
        .css-1d391kg .element-container,
        section[data-testid="stSidebar"] .element-container {
            width: 100% !important;
            max-width: 100% !important;
            padding: 0 1rem;
        }
        
        .css-1d391kg .stSelectbox > div > div > select,
        section[data-testid="stSidebar"] .stSelectbox > div > div > select {
            width: 100% !important;
        }
        
        .css-1d391kg .stSlider > div > div > div,
        section[data-testid="stSidebar"] .stSlider > div > div > div {
            width: 100% !important;
        }
        
        .css-1d391kg .stCheckbox,
        section[data-testid="stSidebar"] .stCheckbox {
            width: 100% !important;
        }
        
        .css-1d391kg .stTextInput > div > div > input,
        section[data-testid="stSidebar"] .stTextInput > div > div > input {
            width: 100% !important;
        }
        
        .css-1d391kg .stTextArea > div > div > textarea,
        section[data-testid="stSidebar"] .stTextArea > div > div > textarea {
            width: 100% !important;
        }
        
        .css-1d391kg .stFileUploader,
        section[data-testid="stSidebar"] .stFileUploader {
            width: 100% !important;
        }
        
        /* Main content adjustment for wide sidebar */
        .main .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1400px;
            margin-left: 0;
        }
        
        /* Force sidebar to stay expanded */
        .css-17eq0hr {
            display: none !important;
        }
        
        /* Override any collapse behavior */
        [data-testid="stSidebarNavItems"] {
            width: 100% !important;
        }
        
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            color: #2d3748;
        }
        
        .stApp {
            background: transparent;
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 3rem 2rem;
            border-radius: 16px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
            pointer-events: none;
        }
        
        .main-header h1 {
            font-size: 3rem;
            font-weight: 800;
            margin: 0;
            line-height: 1.2;
            position: relative;
            z-index: 1;
        }
        
        .main-header p {
            font-size: 1.2rem;
            font-weight: 400;
            margin: 0.5rem 0 0 0;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }
        
        .metric-card {
            background: white;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin-bottom: 1.5rem;
            border-left: 4px solid #667eea;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #667eea, #764ba2);
        }
        
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 30px rgba(0,0,0,0.12);
        }
        
        .metric-card .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #667eea;
            margin: 0.5rem 0;
            line-height: 1;
        }
        
        .metric-card .metric-label {
            font-size: 0.9rem;
            color: #718096;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin: 0;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            text-transform: none;
            letter-spacing: 0.5px;
            width: 100%;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background: white;
            border-radius: 12px;
            padding: 0.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 8px;
            font-weight: 500;
            color: #718096;
            padding: 0.75rem 1.5rem;
            transition: all 0.3s ease;
            border: none;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background: #f7fafc;
            color: #4a5568;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
        }
        
        .dataframe {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
        }
        
        .dataframe th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            border: none;
            padding: 1rem;
        }
        
        .dataframe td {
            border: none;
            padding: 0.75rem 1rem;
            color: #2d3748;
            border-bottom: 1px solid #e2e8f0;
        }
        
        .sidebar-section {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
            width: 100%;
        }
        
        .sidebar-section h3 {
            margin: 0 0 1rem 0;
            color: #2d3748;
            font-weight: 600;
            font-size: 1.1rem;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }
        
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select,
        .stSlider > div > div > div {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            color: #2d3748;
            font-size: 0.95rem;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            outline: none;
        }
        
        .stSuccess {
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 500;
            box-shadow: 0 2px 10px rgba(72, 187, 120, 0.3);
        }
        
        .stError {
            background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 500;
            box-shadow: 0 2px 10px rgba(245, 101, 101, 0.3);
        }
        
        .stWarning {
            background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 500;
            box-shadow: 0 2px 10px rgba(237, 137, 54, 0.3);
        }
        
        .stInfo {
            background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
            border: none;
            border-radius: 12px;
            color: white;
            font-weight: 500;
            box-shadow: 0 2px 10px rgba(66, 153, 225, 0.3);
        }
        
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea, #764ba2);
            border-radius: 8px;
        }
        
        .streamlit-expanderHeader {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            font-weight: 600;
            color: #2d3748;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        
        .plotly-graph-div {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        
        .welcome-section {
            background: white;
            padding: 4rem 2rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            margin: 2rem 0;
        }
        
        .welcome-section h2 {
            color: #2d3748;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .welcome-section p {
            color: #718096;
            font-size: 1.1rem;
            line-height: 1.6;
        }
        
        .footer {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            margin-top: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
        }
        
        .footer p {
            color: #718096;
            margin: 0;
            font-size: 0.9rem;
        }
        
        /* Responsive design */
        @media (max-width: 1200px) {
            .css-1d391kg {
                width: 350px !important;
                min-width: 350px !important;
                max-width: 350px !important;
            }
        }
        
        @media (max-width: 768px) {
            .css-1d391kg {
                width: 100% !important;
                min-width: 100% !important;
                max-width: 100% !important;
            }
            
            .main-header h1 {
                font-size: 2rem;
            }
            
            .main-header p {
                font-size: 1rem;
            }
            
            .metric-card {
                padding: 1.5rem;
            }
            
            .metric-card .metric-value {
                font-size: 2rem;
            }
        }
        
        /* Smooth animations */
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        .fade-in-up {
            animation: fadeInUp 0.6s ease-out;
        }
    </style>
    """

# Load CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'results' not in st.session_state:
    st.session_state.results = {}

# Professional Header
st.markdown("""
<div class="main-header fade-in-up">
    <h1>Pharmaceutical Intelligence Platform</h1>
    <p>Advanced Machine Learning for Drug Classification and Analysis</p>
</div>
""", unsafe_allow_html=True)

# Professional Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section"><h3>Data Configuration</h3></div>', unsafe_allow_html=True)
    
    data_file = st.file_uploader(
        "Upload Pharmaceutical Dataset", 
        type=['xlsx', 'csv'],
        help="Upload your medicines dataset in Excel or CSV format"
    )
    
    if data_file is not None:
        sample_size = st.slider(
            "Sample Size for Analysis", 
            min_value=100, 
            max_value=10000, 
            value=1000, 
            step=100,
            help="Number of records to process (smaller = faster)"
        )
    
    st.markdown('<div class="sidebar-section"><h3>Machine Learning Configuration</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        use_logistic = st.checkbox("Logistic Regression", value=True, help="Fast and interpretable")
        use_svm = st.checkbox("Support Vector Machine", value=True, help="High-performance classification")
    with col2:
        use_nb = st.checkbox("Naive Bayes", value=True, help="Probabilistic predictions")
        use_ensemble = st.checkbox("Ensemble Method", value=False, help="Combine multiple models")
    
    test_size = st.slider(
        "Test Split Percentage", 
        min_value=10, 
        max_value=40, 
        value=20, 
        step=5,
        help="Percentage of data reserved for testing"
    )
    
    st.markdown('<div class="sidebar-section"><h3>System Actions</h3></div>', unsafe_allow_html=True)
    
    if st.button("Initialize System", type="primary", use_container_width=True):
        if data_file is not None:
            with st.spinner("Processing pharmaceutical data..."):
                try:
                    # Load data
                    if data_file.name.endswith('.xlsx'):
                        df = pd.read_excel(data_file)
                    else:
                        df = pd.read_csv(data_file)
                    
                    # Sample data
                    if len(df) > sample_size:
                        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
                    
                    # Basic preprocessing
                    if 'Name' in df.columns and 'Therapeutic Class' in df.columns:
                        df['text'] = df['Name'].fillna('') + ' ' + df['Therapeutic Class'].fillna('')
                        df = df[df['text'].str.strip() != '']
                    
                    st.session_state.data = df
                    st.success(f"Successfully loaded {len(df):,} pharmaceutical records")
                    
                except Exception as e:
                    st.error(f"Data loading failed: {str(e)}")
        else:
            st.warning("Please upload a pharmaceutical dataset first")

# Main content
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Professional Metrics Dashboard
    st.markdown("### System Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card fade-in-up">
            <div class="metric-label">Total Records</div>
            <div class="metric-value">{len(df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'Therapeutic Class' in df.columns:
            classes = df['Therapeutic Class'].nunique()
            st.markdown(f"""
            <div class="metric-card fade-in-up">
                <div class="metric-label">Therapeutic Classes</div>
                <div class="metric-value">{classes}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card fade-in-up">
            <div class="metric-label">Data Features</div>
            <div class="metric-value">{len(df.columns)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        model_count = sum([use_logistic, use_svm, use_nb, use_ensemble])
        st.markdown(f"""
        <div class="metric-card fade-in-up">
            <div class="metric-label">ML Models</div>
            <div class="metric-value">{model_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Professional Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Data Analysis", "Model Training", "Classification", "Performance Metrics"])
    
    with tab1:
        st.markdown("### Pharmaceutical Data Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### Dataset Overview")
            st.dataframe(
                df.head(10), 
                use_container_width=True,
                hide_index=True
            )
            
            if 'Therapeutic Class' in df.columns:
                st.markdown("#### Therapeutic Class Distribution")
                class_counts = df['Therapeutic Class'].value_counts().head(10)
                
                fig = px.bar(
                    x=class_counts.values,
                    y=class_counts.index,
                    orientation='h',
                    title="Top 10 Therapeutic Classes",
                    color=class_counts.values,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    height=500,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#2d3748'),
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Data Statistics")
            
            if 'Therapeutic Class' in df.columns:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Most Common Class</div>
                    <div class="metric-value" style="font-size: 1.5rem;">{df['Therapeutic Class'].mode().iloc[0]}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Class Balance</div>
                    <div class="metric-value" style="font-size: 1.5rem;">{(df['Therapeutic Class'].value_counts().std() / df['Therapeutic Class'].value_counts().mean() * 100):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Data Quality</div>
                <div class="metric-value" style="font-size: 1.5rem;">{df.isnull().sum().sum() == 0 and 'Complete' or 'Incomplete'}</div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### Machine Learning Model Training")
        
        if st.button("Train Classification Models", type="primary", use_container_width=True):
            if 'text' in df.columns and 'Therapeutic Class' in df.columns:
                with st.spinner("Training pharmaceutical classification models..."):
                    try:
                        # Prepare data
                        X = df['text']
                        y = df['Therapeutic Class']
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42, stratify=y
                        )
                        
                        # Vectorize text
                        vectorizer = TfidfVectorizer(
                            max_features=5000,
                            stop_words='english',
                            ngram_range=(1, 2),
                            min_df=2,
                            max_df=0.8
                        )
                        
                        X_train_vec = vectorizer.fit_transform(X_train)
                        X_test_vec = vectorizer.transform(X_test)
                        
                        st.session_state.vectorizer = vectorizer
                        
                        # Train models
                        models = {}
                        results = {}
                        
                        if use_logistic:
                            lr = LogisticRegression(
                                max_iter=1000, 
                                random_state=42, 
                                n_jobs=-1,
                                C=1.0
                            )
                            lr.fit(X_train_vec, y_train)
                            lr_pred = lr.predict(X_test_vec)
                            lr_acc = accuracy_score(y_test, lr_pred)
                            models['Logistic Regression'] = lr
                            results['Logistic Regression'] = lr_acc
                        
                        if use_svm:
                            svm = LinearSVC(
                                max_iter=2000, 
                                random_state=42,
                                C=1.0
                            )
                            svm.fit(X_train_vec, y_train)
                            svm_pred = svm.predict(X_test_vec)
                            svm_acc = accuracy_score(y_test, svm_pred)
                            models['Support Vector Machine'] = svm
                            results['Support Vector Machine'] = svm_acc
                        
                        if use_nb:
                            nb = MultinomialNB(alpha=1.0)
                            nb.fit(X_train_vec, y_train)
                            nb_pred = nb.predict(X_test_vec)
                            nb_acc = accuracy_score(y_test, nb_pred)
                            models['Naive Bayes'] = nb
                            results['Naive Bayes'] = nb_acc
                        
                        st.session_state.models = models
                        st.session_state.results = results
                        
                        # Display training results
                        st.success("Machine learning models trained successfully!")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        for i, (model_name, accuracy) in enumerate(results.items()):
                            with [col1, col2, col3][i % 3]:
                                st.markdown(f"""
                                <div class="metric-card fade-in-up">
                                    <div class="metric-label">{model_name}</div>
                                    <div class="metric-value">{accuracy:.3f}</div>
                                </div>
                                """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Model training failed: {str(e)}")
            else:
                st.warning("Dataset must contain 'text' and 'Therapeutic Class' columns")
    
    with tab3:
        st.markdown("### Pharmaceutical Classification")
        
        if st.session_state.models:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                medicine_input = st.text_input(
                    "Enter Medicine Name for Classification",
                    placeholder="e.g., Paracetamol, Aspirin, Ibuprofen",
                    help="Type the name of a pharmaceutical product to classify"
                )
                
                if st.button("Classify Medicine", type="primary", use_container_width=True):
                    if medicine_input and st.session_state.vectorizer:
                        with st.spinner("Analyzing pharmaceutical data..."):
                            try:
                                # Predict with all models
                                X_input = st.session_state.vectorizer.transform([medicine_input])
                                
                                st.markdown("### Classification Results")
                                
                                for model_name, model in st.session_state.models.items():
                                    prediction = model.predict(X_input)[0]
                                    confidence = max(model.predict_proba(X_input)[0]) if hasattr(model, 'predict_proba') else "N/A"
                                    
                                    st.markdown(f"""
                                    <div class="metric-card fade-in-up">
                                        <div class="metric-label">{model_name}</div>
                                        <div class="metric-value">{prediction}</div>
                                        {f'<div style="font-size: 0.9rem; color: #718096; margin-top: 0.5rem;">Confidence: {confidence:.3f}</div>' if confidence != "N/A" else ''}
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            except Exception as e:
                                st.error(f"Classification failed: {str(e)}")
                    else:
                        st.warning("Please enter a medicine name")
            
            with col2:
                st.markdown("#### Classification Guidelines")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Available Models</div>
                    <div class="metric-value" style="font-size: 1.2rem;">{len(st.session_state.models)}</div>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Feature Space</div>
                    <div class="metric-value" style="font-size: 1.2rem;">{st.session_state.vectorizer.get_feature_names_out().shape[0]:,}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Please train machine learning models first in the Model Training tab")
    
    with tab4:
        st.markdown("### Model Performance Analysis")
        
        if st.session_state.results:
            results = st.session_state.results
            
            # Performance comparison chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(results.keys()),
                    y=list(results.values()),
                    marker=dict(
                        color=['#667eea', '#764ba2', '#48bb78', '#ed8936'][:len(results)],
                        line=dict(color='white', width=2)
                    ),
                    text=[f'{acc:.3f}' for acc in results.values()],
                    textposition='auto',
                    textfont=dict(color='white', size=12),
                    hovertemplate='<b>%{x}</b><br>Accuracy: %{y:.3f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Model Accuracy Comparison",
                xaxis_title="Machine Learning Models",
                yaxis_title="Accuracy Score",
                yaxis=dict(range=[0, 1], tickformat='.0%'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#2d3748', size=12),
                margin=dict(l=20, r=20, t=40, b=20),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            st.markdown("#### Detailed Model Metrics")
            
            for model_name, model in st.session_state.models.items():
                with st.expander(f"{model_name} Performance Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Accuracy", f"{results[model_name]:.3f}")
                        st.metric("Model Type", "Classification")
                    
                    with col2:
                        st.metric("Training Status", "Completed")
                        st.metric("Features Used", st.session_state.vectorizer.get_feature_names_out().shape[0])
        else:
            st.info("No performance metrics available. Please train models first.")

else:
    st.markdown("""
    <div class="welcome-section fade-in-up">
        <h2>Welcome to Pharmaceutical Intelligence Platform</h2>
        <p>Upload your pharmaceutical dataset in the sidebar to begin advanced machine learning analysis.</p>
        <p>Your dataset should include 'Name' and 'Therapeutic Class' columns for optimal results.</p>
    </div>
    """, unsafe_allow_html=True)

# Professional Footer
st.markdown("""
<div class="footer">
    <p>Pharmaceutical Intelligence Platform © 2024 | Advanced ML-Powered Drug Analysis System</p>
    <p>Built with Streamlit, scikit-learn, and Modern Web Technologies</p>
</div>
""", unsafe_allow_html=True)
