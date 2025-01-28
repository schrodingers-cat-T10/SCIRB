# SCIRB - schrödinger's cat in radiation box (what if - explainable AI)
# Capital Project Management and Prediction Dashboard

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Cohere](https://img.shields.io/badge/Cohere-FFFFFF?style=for-the-badge&logo=Cohere&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=Plotly&logoColor=white)

Welcome to the **Capital Project Management and Prediction Dashboard**! This project is a comprehensive Streamlit-based web application designed to analyze, visualize, and predict outcomes for capital projects. It integrates machine learning models, data visualization tools, and an AI-powered chatbot to provide actionable insights.

---

## Features

### 🏠 **Dashboard**
- **Interactive Visualizations**: Explore project data through dynamic charts and graphs.
  - Projects by Borough
  - Budget Allocation Over Time
  - Project Delays
  - Budget Distribution by Project Type
  - City vs Non-City Budget Allocation
  - Feature Correlation Heatmap
- **File Upload**: Upload your dataset (CSV) to analyze and visualize project data.

### 📊 **Prediction**
- **Machine Learning Models**:
  - **Linear Model**: Predict `CITY_PLAN_TOTAL`.
  - **Categorical Model**: Predict `DELAY_DESC` (project delays).
  - **Regression Model**: Predict 5-Year Plan budgets.
- **LIME Explanations**: Understand model predictions with Local Interpretable Model-agnostic Explanations (LIME).

### 🤖 **Chatbot**
- **AI-Powered Assistant**: Interact with a Cohere-powered chatbot to get answers to your queries.
- **Python REPL Integration**: Execute Python code directly within the chatbot.
- **Custom Tools**: Access project data and perform advanced queries.

---

## Technologies Used

- **Frontend**: Streamlit
- **Backend**: TensorFlow, Scikit-learn, Pandas, NumPy
- **Visualization**: Plotly
- **AI Chatbot**: Cohere, LangChain
- **Model Interpretability**: LIME

---

## Installation

To run this project locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/capital-project-dashboard.git
   cd capital-project-dashboard
