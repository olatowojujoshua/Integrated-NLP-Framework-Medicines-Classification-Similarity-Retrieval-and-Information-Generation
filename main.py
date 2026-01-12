import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os

# Page Configuration
st.set_page_config(
    page_title="Medicines Analysis Platform",
    page_icon="Pill",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
def load_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
        }
        
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
            border-left: 4px solid #667eea;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: 500;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 5px;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .dataframe {
            border-radius: 10px;
            overflow: hidden;
        }
        
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
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

# Header
st.markdown("""
<div class="main-header">
    <h1>Medicines Analysis Platform</h1>
    <p>Advanced Machine Learning for Pharmaceutical Classification</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-section"><h3>Data Configuration</h3></div>', unsafe_allow_html=True)
    
    data_file = st.file_uploader("Upload Medicine Dataset", type=['xlsx', 'csv'])
    sample_size = st.slider("Sample Size", 100, 10000, 1000)
    
    st.markdown('<div class="sidebar-section"><h3>Model Settings</h3></div>', unsafe_allow_html=True)
    
    use_logistic = st.checkbox("Logistic Regression", value=True)
    use_svm = st.checkbox("SVM", value=True)
    use_nb = st.checkbox("Naive Bayes", value=True)
    
    test_size = st.slider("Test Size (%)", 10, 40, 20)
    
    if st.button("Initialize System", type="primary"):
        if data_file is not None:
            with st.spinner("Loading and processing data..."):
                try:
                    # Load data
                    if data_file.name.endswith('.xlsx'):
                        df = pd.read_excel(data_file)
                    else:
                        df = pd.read_csv(data_file)
                    
                    # Sample data
                    if len(df) > sample_size:
                        df = df.sample(n=sample_size, random_state=42)
                    
                    # Basic preprocessing
                    if 'Name' in df.columns and 'Therapeutic Class' in df.columns:
                        df['text'] = df['Name'].fillna('') + ' ' + df['Therapeutic Class'].fillna('')
                        df = df[df['text'].str.strip() != '']
                    
                    st.session_state.data = df
                    st.success(f"Data loaded: {len(df)} records")
                    
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        else:
            st.warning("Please upload a dataset first")

# Main content
if st.session_state.data is not None:
    df = st.session_state.data
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Records</h3>
            <h2>{len(df):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if 'Therapeutic Class' in df.columns:
            classes = df['Therapeutic Class'].nunique()
            st.markdown(f"""
            <div class="metric-card">
                <h3>Classes</h3>
                <h2>{classes}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Features</h3>
            <h2>{len(df.columns)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Models</h3>
            <h2>{sum([use_logistic, use_svm, use_nb])}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🤖 Models", "🔍 Analysis", "📈 Results"])
    
    with tab1:
        st.markdown("### Data Overview")
        st.dataframe(df.head(10), use_container_width=True)
        
        if 'Therapeutic Class' in df.columns:
            st.markdown("### Class Distribution")
            class_counts = df['Therapeutic Class'].value_counts().head(10)
            
            fig = px.bar(
                x=class_counts.values,
                y=class_counts.index,
                orientation='h',
                title="Top 10 Therapeutic Classes"
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### Model Training")
        
        if st.button("Train Models", type="primary"):
            if 'text' in df.columns and 'Therapeutic Class' in df.columns:
                with st.spinner("Training models..."):
                    try:
                        # Prepare data
                        X = df['text']
                        y = df['Therapeutic Class']
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size/100, random_state=42
                        )
                        
                        # Vectorize
                        vectorizer = TfidfVectorizer(
                            max_features=5000,
                            stop_words='english',
                            ngram_range=(1, 2)
                        )
                        X_train_vec = vectorizer.fit_transform(X_train)
                        X_test_vec = vectorizer.transform(X_test)
                        
                        st.session_state.vectorizer = vectorizer
                        
                        # Train models
                        models = {}
                        results = {}
                        
                        if use_logistic:
                            lr = LogisticRegression(max_iter=1000, random_state=42)
                            lr.fit(X_train_vec, y_train)
                            lr_pred = lr.predict(X_test_vec)
                            lr_acc = accuracy_score(y_test, lr_pred)
                            models['Logistic Regression'] = lr
                            results['Logistic Regression'] = lr_acc
                        
                        if use_svm:
                            svm = LinearSVC(max_iter=1000, random_state=42)
                            svm.fit(X_train_vec, y_train)
                            svm_pred = svm.predict(X_test_vec)
                            svm_acc = accuracy_score(y_test, svm_pred)
                            models['SVM'] = svm
                            results['SVM'] = svm_acc
                        
                        if use_nb:
                            nb = MultinomialNB()
                            nb.fit(X_train_vec, y_train)
                            nb_pred = nb.predict(X_test_vec)
                            nb_acc = accuracy_score(y_test, nb_pred)
                            models['Naive Bayes'] = nb
                            results['Naive Bayes'] = nb_acc
                        
                        st.session_state.models = models
                        st.session_state.results = results
                        
                        # Display results
                        st.success("Models trained successfully!")
                        
                        for model_name, accuracy in results.items():
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{model_name}</h3>
                                <h2>{accuracy:.3f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    except Exception as e:
                        st.error(f"Training error: {str(e)}")
            else:
                st.warning("Please ensure data has 'text' and 'Therapeutic Class' columns")
    
    with tab3:
        st.markdown("### Medicine Classification")
        
        if st.session_state.models:
            medicine_input = st.text_input("Enter Medicine Name", placeholder="e.g., Paracetamol")
            
            if st.button("Classify", type="primary"):
                if medicine_input and st.session_state.vectorizer:
                    with st.spinner("Classifying..."):
                        try:
                            # Predict with all models
                            X_input = st.session_state.vectorizer.transform([medicine_input])
                            
                            st.markdown("### Classification Results")
                            
                            for model_name, model in st.session_state.models.items():
                                prediction = model.predict(X_input)[0]
                                st.markdown(f"""
                                <div class="metric-card">
                                    <h3>{model_name}</h3>
                                    <h2>{prediction}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        except Exception as e:
                            st.error(f"Classification error: {str(e)}")
        else:
            st.info("Please train models first")
    
    with tab4:
        st.markdown("### Model Performance")
        
        if 'results' in st.session_state:
            results = st.session_state.results
            
            # Performance chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(results.keys()),
                    y=list(results.values()),
                    marker_color='rgb(102, 126, 234)',
                    text=[f'{acc:.3f}' for acc in results.values()],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Model Accuracy Comparison",
                xaxis_title="Models",
                yaxis_title="Accuracy",
                yaxis=dict(range=[0, 1])
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed metrics
            st.markdown("### Detailed Metrics")
            
            for model_name, model in st.session_state.models.items():
                with st.expander(f"{model_name} Details"):
                    st.write(f"Accuracy: {results[model_name]:.3f}")
                    st.write("Model trained successfully")
        else:
            st.info("No results available yet")

else:
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>Welcome to Medicines Analysis Platform</h2>
        <p>Please upload your dataset in the sidebar to get started.</p>
        <p>Your dataset should include 'Name' and 'Therapeutic Class' columns.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Medicines Analysis Platform © 2024 | Built with Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)
