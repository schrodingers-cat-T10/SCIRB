import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.metrics import MeanAbsoluteError
import lime
from langchain_experimental.tools.python.tool import PythonAstREPLTool
import lime.lime_tabular  # For LIME visualization
import plotly.express as px  # For visualizations
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_cohere.chat_models import ChatCohere
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_experimental.utilities import PythonREPL
from langchain.agents import Tool
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
import os
import pandas as pd

# Load the saved models with custom objects
linear_model = load_model("linear.h5", custom_objects={"mae": MeanAbsoluteError()})
cat_model = load_model("cat.h5", custom_objects={"mae": MeanAbsoluteError()})
regression_model = load_model("regressions1.h5", custom_objects={"mae": MeanAbsoluteError()})

# Function to preprocess input data for each model
def preprocess_input(data, model_type):
    categorical_columns = ["BORO", "MANAGING_AGCY", "PROJECT_DESCR", "PROJECT_ID", "COMMUNITY_BOARD",
                           "TYP_CATEGORY_NAME", "BUDGET_LINE", "DELAY_DESC", "SITE_DESCR"]
    encoder = LabelEncoder()
    for col in categorical_columns:
        if col in data.columns:
            data[col] = encoder.fit_transform(data[col])

    # Date processing (if applicable)
    if "PUB_DATE" in data.columns:  # Assuming PUB_DATE is not in the provided columns
        data["PUB_DATE"] = pd.to_datetime(data["PUB_DATE"], format="%Y%m%d")
        data["month"] = data["PUB_DATE"].dt.month
        data["day"] = data["PUB_DATE"].dt.day
        data["year"] = data["PUB_DATE"].dt.year
        data = data.drop("PUB_DATE", axis=1)

    xdata = data

    # Standard scaling
    scaler = StandardScaler()
    xdata = scaler.fit_transform(xdata)

    # Reshape for LSTM: (batch_size, timesteps, features)
    xdata = np.array(xdata).reshape((xdata.shape[0], 1, xdata.shape[1]))

    return xdata, scaler

# Custom CSS for sidebar styling
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content .block-container {
        padding: 1rem;
    }
    .sidebar .sidebar-content .stRadio > div {
        flex-direction: column;
    }
    .sidebar .sidebar-content .stRadio label {
        margin: 0.5rem 0;
        padding: 0.5rem;
        border-radius: 0.5rem;
        transition: background-color 0.3s;
    }
    .sidebar .sidebar-content .stRadio label:hover {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Dashboard", "📊 Prediction", "🤖 Chatbot"],
    key="navigation"
)

if page == "🏠 Dashboard":
    st.title("Dashboard")
    st.write("Welcome to the Dashboard page! Explore insights and trends in project data.")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Load the dataset
        data = pd.read_csv(uploaded_file)

        # Add month and year columns for time-based analysis
        data["PUB_DATE"] = pd.to_datetime(data["PUB_DATE"], format="%Y%m%d")
        data["month"] = data["PUB_DATE"].dt.month
        data["year"] = data["PUB_DATE"].dt.year

        # Dropdown for visualization selection
        visualization_option = st.selectbox(
            "Choose a Visualization",
            [
                "Projects by Borough",
                "Budget Allocation Over Time",
                "Project Delays",
                "Budget Distribution by Project Type",
                "City vs Non-City Budget Allocation",
                "Feature Correlation Heatmap"
            ],
            key="visualization_selectbox"
        )

        # Key Visualizations

        # 1. Projects by Borough (Bar Chart)
        if visualization_option == "Projects by Borough":
            st.subheader("Projects by Borough")
            borough_counts = data["BORO"].value_counts().reset_index()
            borough_counts.columns = ["Borough", "Number of Projects"]
            fig1 = px.bar(borough_counts, x="Borough", y="Number of Projects", color="Borough", title="Number of Projects by Borough")
            st.plotly_chart(fig1, use_container_width=True)

        # 2. Budget Allocation Over Time (Line Chart)
        elif visualization_option == "Budget Allocation Over Time":
            st.subheader("Budget Allocation Over Time")
            budget_over_time = data.groupby("year")["ORIG_BUD_AMT"].sum().reset_index()
            fig2 = px.line(budget_over_time, x="year", y="ORIG_BUD_AMT", title="Budget Allocation Over Time")
            st.plotly_chart(fig2, use_container_width=True)

        # 3. Project Delays (Pie Chart)
        elif visualization_option == "Project Delays":
            st.subheader("Project Delays")
            delay_counts = data["DELAY_DESC"].value_counts().reset_index()
            delay_counts.columns = ["Status", "Count"]
            fig3 = px.pie(delay_counts, values="Count", names="Status", title="Percentage of Delayed vs On-Time Projects")
            st.plotly_chart(fig3, use_container_width=True)

        # 4. Budget Distribution by Project Type (Box Plot)
        elif visualization_option == "Budget Distribution by Project Type":
            st.subheader("Budget Distribution by Project Type")
            fig4 = px.box(data, x="TYP_CATEGORY_NAME", y="ORIG_BUD_AMT", color="TYP_CATEGORY_NAME", title="Budget Distribution by Project Type")
            st.plotly_chart(fig4, use_container_width=True)

        # 5. City vs Non-City Budget Allocation (Bar Chart)
        elif visualization_option == "City vs Non-City Budget Allocation":
            st.subheader("City vs Non-City Budget Allocation")
            city_budget = data["CITY_PLAN_TOTAL"].sum()
            non_city_budget = data["NONCITY_PLAN_TOTAL"].sum()
            budget_data = pd.DataFrame({
                "Budget Type": ["City Budget", "Non-City Budget"],
                "Total Budget": [city_budget, non_city_budget]
            })
            fig5 = px.bar(budget_data, x="Budget Type", y="Total Budget", color="Budget Type", title="City vs Non-City Budget Allocation")
            st.plotly_chart(fig5, use_container_width=True)

        # 6. Correlation Heatmap
        elif visualization_option == "Feature Correlation Heatmap":
            st.subheader("Feature Correlation Heatmap")
            corr = data[["ORIG_BUD_AMT", "CITY_PLAN_TOTAL", "NONCITY_PLAN_TOTAL", "CITY_PRIOR_ACTUAL", "NONCITY_PRIOR_ACTUAL"]].corr()
            fig6 = px.imshow(corr, text_auto=True, title="Correlation Between Financial Features")
            st.plotly_chart(fig6, use_container_width=True)

    else:
        st.info("Please upload a CSV file to get started.")

elif page == "📊 Prediction":
    st.title("Capital Project Prediction")
    st.markdown("""
        <style>
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
        .stSelectbox, .stNumberInput, .stTextInput {
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Task selection
    task = st.selectbox("Select Task", ["Linear (Predict CITY_PLAN_TOTAL)", "Cat (Predict DELAY_DESC)", "Regression (Predict 5-Year Plan)"])

    # Initialize an empty dictionary to store user inputs
    input_data = {}

    # Common input fields
    st.header("Input Features")
    with st.expander("General Information", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            input_data["BORO"] = st.text_input("BORO", value="Manhattan", help="Enter the borough (e.g., Manhattan, Brooklyn).")
            input_data["MANAGING_AGCY_CD"] = st.number_input("MANAGING_AGCY_CD", value=0, help="Enter the managing agency code.")
            input_data["MANAGING_AGCY"] = st.text_input("MANAGING_AGCY", value="Agency Name", help="Enter the managing agency.")
            input_data["PROJECT_ID"] = st.text_input("PROJECT_ID", value="Project123", help="Enter the project ID.")
            input_data["PROJECT_DESCR"] = st.text_input("PROJECT_DESCR", value="Road Construction", help="Enter the project description.")

        with col2:
            input_data["TYP_CATEGORY_NAME"] = st.text_input("TYP_CATEGORY_NAME", value="Infrastructure", help="Enter the type category name.")
            input_data["COMMUNITY_BOARD"] = st.text_input("COMMUNITY_BOARD", value="CB1", help="Enter the community board.")
            input_data["BUDGET_LINE"] = st.text_input("BUDGET_LINE", value="BL123", help="Enter the budget line.")
            input_data["SITE_DESCR"] = st.text_input("SITE_DESCR", value="Site Description", help="Enter the site description.")

    with st.expander("Financial Information", expanded=True):
        col3, col4 = st.columns(2)

        with col3:
            input_data["FY_YR1_PLAN"] = st.number_input("FY_YR1_PLAN", value=0, help="Enter the FY Year 1 plan.")
            input_data["ORIG_BUD_AMT"] = st.number_input("ORIG_BUD_AMT", value=0, help="Enter the original budget amount.")
            input_data["CITY_PRIOR_ACTUAL"] = st.number_input("CITY_PRIOR_ACTUAL", value=0.0, help="Enter the city prior actual.")

        with col4:
            input_data["CITY_RTC"] = st.number_input("CITY_RTC", value=0, help="Enter the city RTC.")
            input_data["month"] = st.number_input("Month", value=1, min_value=1, max_value=12, help="Enter the month.")
            input_data["day"] = st.number_input("Day", value=1, min_value=1, max_value=31, help="Enter the day.")
            input_data["year"] = st.number_input("Year", value=2023, help="Enter the year.")

    # Task-specific input fields
    if task == "Linear (Predict CITY_PLAN_TOTAL)":
        with st.expander("Linear Prediction Details", expanded=True):
            input_data["DELAY_DESC"] = st.text_input("DELAY_DESC", value="No Delay", help="Enter the delay description.")
            input_data["CITY_YR1_PLAN"] = st.number_input("CITY_YR1_PLAN", value=0, help="Enter the city year 1 plan.")
            input_data["CITY_YR2_PLAN"] = st.number_input("CITY_YR2_PLAN", value=0, help="Enter the city year 2 plan.")
            input_data["CITY_YR3_PLAN"] = st.number_input("CITY_YR3_PLAN", value=0, help="Enter the city year 3 plan.")
            input_data["CITY_YR4_PLAN"] = st.number_input("CITY_YR4_PLAN", value=0, help="Enter the city year 4 plan.")
            input_data["CITY_YR5_PLAN"] = st.number_input("CITY_YR5_PLAN", value=0, help="Enter the city year 5 plan.")
    elif task == "Cat (Predict DELAY_DESC)":
        with st.expander("Categorical Prediction Details", expanded=True):
            input_data["CITY_PLAN_TOTAL"] = st.number_input("CITY_PLAN_TOTAL", value=0.0, help="Enter the city plan total.")
            input_data["CITY_YR1_PLAN"] = st.number_input("CITY_YR1_PLAN", value=0, help="Enter the city year 1 plan.")
            input_data["CITY_YR2_PLAN"] = st.number_input("CITY_YR2_PLAN", value=0, help="Enter the city year 2 plan.")
            input_data["CITY_YR3_PLAN"] = st.number_input("CITY_YR3_PLAN", value=0, help="Enter the city year 3 plan.")
            input_data["CITY_YR4_PLAN"] = st.number_input("CITY_YR4_PLAN", value=0, help="Enter the city year 4 plan.")
            input_data["CITY_YR5_PLAN"] = st.number_input("CITY_YR5_PLAN", value=0, help="Enter the city year 5 plan.")
    elif task == "Regression (Predict 5-Year Plan)":
        with st.expander("Regression Prediction Details", expanded=True):
            input_data["CITY_PLAN_TOTAL"] = st.number_input("CITY_PLAN_TOTAL", value=0.0, help="Enter the city plan total.")
            input_data["DELAY_DESC"] = st.text_input("DELAY_DESC", value="No Delay", help="Enter the delay description.")

    # Create a DataFrame from user input
    input_data = pd.DataFrame([input_data])

    # Buttons for predictions
    if st.button("Predict", key="predict_button"):
        if task == "Linear (Predict CITY_PLAN_TOTAL)":
            xdata, scaler = preprocess_input(input_data, model_type="linear")
            if xdata is not None:
                prediction = linear_model.predict(xdata)
                st.success(f"*Predicted CITY_PLAN_TOTAL:* {prediction[0][0]:.2f}")

                # LIME Explanation
                st.subheader("LIME Explanation")
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.zeros((1, xdata.shape[2])),  # Dummy data for LIME
                    mode="regression",
                    feature_names=input_data.columns,
                    verbose=True,
                    discretize_continuous=True
                )
                exp = explainer.explain_instance(
                    xdata[0].flatten(),  # Flatten the input for LIME
                    lambda x: linear_model.predict(x.reshape(-1, 1, xdata.shape[2])),
                    num_features=len(input_data.columns)
                )
                st.pyplot(exp.as_pyplot_figure())

        elif task == "Cat (Predict DELAY_DESC)":
            xdata, scaler = preprocess_input(input_data, model_type="cat")
            if xdata is not None:
                prediction = cat_model.predict(xdata)
                result = "Delay" if prediction[0][0] > 0.5 else "No Delay"
                st.success(f"*Predicted DELAY_DESC:* {result}")

                # LIME Explanation
                st.subheader("LIME Explanation")
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.zeros((1, xdata.shape[2])),  # Dummy data for LIME
                    mode="classification",
                    feature_names=input_data.columns,
                    verbose=True,
                    discretize_continuous=True
                )
                exp = explainer.explain_instance(
                    xdata[0].flatten(),  # Flatten the input for LIME
                    lambda x: cat_model.predict(x.reshape(-1, 1, xdata.shape[2])),
                    num_features=len(input_data.columns)
                )
                st.pyplot(exp.as_pyplot_figure())

        elif task == "Regression (Predict 5-Year Plan)":
            xdata, scaler = preprocess_input(input_data, model_type="regression")
            if xdata is not None:
                prediction = regression_model.predict(xdata)
                st.success("*Predicted 5-Year Plan:*")
                for i, value in enumerate(prediction[0], start=1):
                    st.write(f"Year {i}: {value:.2f}")

                # LIME Explanation
                st.subheader("LIME Explanation")
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=np.zeros((1, xdata.shape[2])),  # Dummy data for LIME
                    mode="regression",
                    feature_names=input_data.columns,
                    verbose=True,
                    discretize_continuous=True
                )
                exp = explainer.explain_instance(
                    xdata[0].flatten(),  # Flatten the input for LIME
                    lambda x: regression_model.predict(x.reshape(-1, 1, xdata.shape[2])),
                    num_features=len(input_data.columns)
                )
                st.pyplot(exp.as_pyplot_figure())

elif page == "🤖 Chatbot":
    st.title("Chatbot")
    st.write("Welcome to the Chatbot page!")

    # Add a text input for user queries
    user_input = st.text_input("Enter your query:", placeholder="Ask me anything...")

    # Add a button to submit the query
    if st.button("Submit"):
        if user_input.strip() == "":
            st.warning("Please enter a query.")
        else:
            os.environ['COHERE_API_KEY'] = "yTmKdlP6vaGOZ91YAlPCKqMUpvmD2rgSoZqZJRHS"

            # Initialize the chat model
            chat = ChatCohere(model="command-r-plus", temperature=0.3)
            prompt = ChatPromptTemplate.from_template("{input}")

            # Set up the Python REPL tool
            python_repl = PythonREPL()
            repl_tool = Tool(
                name="python_repl",
                description="Executes python code and returns the result. The code runs in a static sandbox without interactive mode, so print output or save output to a file.",
                func=python_repl.run,
            )

            df = pd.read_csv("hello.csv")
            tool = PythonAstREPLTool(locals={"df": df},description="you can convert the csv file into pandas dataframe and return")


            track_tool=Tool(
                name="tracker_tool",
                description="you can convert the csv file into pandas dataframe and return the csv file will be named as hello.csv",
                func=tool.run,
            )

            # Create the Cohere React agent
            agent = create_cohere_react_agent(
                llm=chat,
                tools=[repl_tool,tool],
                prompt=prompt,
            )

            # Initialize the agent executor
            agent_executor = AgentExecutor(agent=agent, tools=[repl_tool,tool], verbose=True)

            # Invoke the agent with the user's input
            try:
                result = agent_executor.invoke({"input": user_input})
                st.success("Chatbot Response:")
                st.write(result["output"])
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")