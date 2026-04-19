import streamlit as st
import pandas as pd
import numpy as np
from groq_client import get_llm_response, clean_json, generate_insights
import json 
import seaborn as sns
import matplotlib.pyplot as plt
from difflib import get_close_matches
import traceback

st.set_page_config(page_title="AI Data Analyst", layout="centered")

st.title("🧠 DataSage AI")
st.markdown("AI-powered data analyst that understands natural language and performs analysis instantly.")

# Initialize session state for dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

def resolve_column(user_column, df_columns):
    """Resolve column name with fuzzy matching"""
    if user_column is None or pd.isna(user_column):
        return None
    
    # Exact match
    if user_column in df_columns:
        return user_column
    
    # Case-insensitive match
    for col in df_columns:
        if col.lower() == user_column.lower():
            return col
    
    # Fuzzy match
    matches = get_close_matches(user_column, df_columns, n=1, cutoff=0.6)
    if matches:
        return matches[0]
    
    return None

def safe_execute_operation(df, operation, column, groupby_col=None, value=None):
    """Safely execute data operations with error handling"""
    try:
        if column is None:
            return None, "No column specified"
        
        # Ensure column exists in dataframe
        if column not in df.columns:
            return None, f"Column '{column}' not found in dataframe"
        
        # Check if column is numeric for numeric operations
        numeric_ops = ['mean', 'sum', 'max', 'min', 'percentile', 'iqr']
        if operation in numeric_ops and not pd.api.types.is_numeric_dtype(df[column]):
            return None, f"Column '{column}' must be numeric for {operation} operation"
        
        # Perform operation
        if operation == "mean":
            if groupby_col and groupby_col in df.columns:
                result = df.groupby(groupby_col)[column].mean().round(3)
                return result, None
            else:
                result = round(df[column].mean(), 3)
                return result, None
        
        elif operation == "sum":
            if groupby_col and groupby_col in df.columns:
                result = df.groupby(groupby_col)[column].sum()
                return result, None
            else:
                result = df[column].sum()
                return result, None
        
        elif operation == "max":
            if groupby_col and groupby_col in df.columns:
                result = df.groupby(groupby_col)[column].max()
                return result, None
            else:
                result = df[column].max()
                return result, None
        
        elif operation == "min":
            if groupby_col and groupby_col in df.columns:
                result = df.groupby(groupby_col)[column].min()
                return result, None
            else:
                result = df[column].min()
                return result, None
        
        elif operation == "count":
            if groupby_col and groupby_col in df.columns:
                result = df.groupby(groupby_col)[column].count()
                return result, None
            else:
                result = df[column].count()
                return result, None
        
        elif operation == "percentile":
            percentile_value = value if value else 50
            result = df[column].quantile(percentile_value / 100)
            return result, None
        
        elif operation == "iqr":
            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)
            result = q3 - q1
            return result, None
        
        else:
            return None, f"Operation '{operation}' not supported"
            
    except Exception as e:
        return None, f"Error in {operation}: {str(e)}"

def create_plot(df, plot_type, x_col, y_col=None):
    """Create plots with proper error handling"""
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if plot_type == "histogram":
            if x_col and x_col in df.columns:
                sns.histplot(df[x_col].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {x_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel("Frequency")
            else:
                st.error(f"Column '{x_col}' not found for histogram")
                return None
                
        elif plot_type == "box":
            if x_col and x_col in df.columns:
                sns.boxplot(y=df[x_col].dropna(), ax=ax)
                ax.set_title(f"Box Plot of {x_col}")
                ax.set_ylabel(x_col)
            else:
                st.error(f"Column '{x_col}' not found for box plot")
                return None
                
        elif plot_type == "bar":
            if x_col and x_col in df.columns:
                if y_col and y_col in df.columns:
                    # Bar chart with two columns
                    ax.bar(df[x_col].head(20), df[y_col].head(20))
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                else:
                    # Count plot for single column
                    value_counts = df[x_col].value_counts().head(20)
                    ax.bar(range(len(value_counts)), value_counts.values)
                    ax.set_xticks(range(len(value_counts)))
                    ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel("Count")
                ax.set_title(f"Bar Chart of {x_col}" + (f" vs {y_col}" if y_col else ""))
            else:
                st.error(f"Column '{x_col}' not found for bar chart")
                return None
                
        elif plot_type == "line":
            if x_col and x_col in df.columns and y_col and y_col in df.columns:
                ax.plot(df[x_col].head(100), df[y_col].head(100), marker='o', linewidth=2, markersize=4)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"Line Plot: {x_col} vs {y_col}")
                ax.grid(True, alpha=0.3)
            else:
                st.error(f"Columns not found for line plot. Need both x and y columns")
                return None
                
        elif plot_type == "scatter":
            if x_col and x_col in df.columns and y_col and y_col in df.columns:
                ax.scatter(df[x_col].head(500), df[y_col].head(500), alpha=0.6)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
                ax.grid(True, alpha=0.3)
            else:
                st.error(f"Columns not found for scatter plot. Need both x and y columns")
                return None
        
        elif plot_type == "correlation":
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] >= 2:
                corr_matrix = numeric_df.corr()
                sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax, fmt='.2f', square=True)
                ax.set_title("Correlation Matrix")
            else:
                st.error("Need at least 2 numeric columns for correlation matrix")
                return None
        else:
            st.error(f"Plot type '{plot_type}' not supported")
            return None
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return None

# File upload
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        df = st.session_state.df
        
        st.success("✅ Dataset loaded successfully!")
        
        # Show basic info in columns
        st.subheader("📌 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Total Values", df.size)
        col4.metric("Missing Values", df.isnull().sum().sum())
        
        # Column info
        with st.expander("📋 Column Information", expanded=True):
            col_info = pd.DataFrame({
                'Type': df.dtypes.astype(str),
                'Non-Null': df.notnull().sum(),
                'Null %': (df.isnull().sum() / len(df) * 100).round(2),
                'Unique': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Data preview
        with st.expander("👀 Data Preview (First 10 rows)"):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Statistical summary
        with st.expander("📈 Statistical Summary"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns found for statistical summary")
        
        # Missing values
        with st.expander("⚠️ Missing Values Details"):
            missing_df = pd.DataFrame({
                'Missing Count': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if len(missing_df) > 0:
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("No missing values found!")
        
        # Auto analyze button
        st.subheader("🤖 AI-Powered Analysis")
        col1, col2 = st.columns([1, 3])
        
        with col1:
            if st.button("⚡ Auto Analyze Dataset", use_container_width=True):
                with st.spinner("Generating insights..."):
                    insights = generate_insights(df)
                    st.session_state.insights = insights
        
        if 'insights' in st.session_state:
            with st.expander("🧠 AI-Generated Insights", expanded=True):
                st.markdown(st.session_state.insights)
        
        # Query input
        st.subheader("💬 Ask Questions About Your Data")
        
        # Example queries
        with st.expander("📝 Example Queries"):
            st.markdown("""
            - What is the average age?
            - Show me the sum of sales by category
            - Plot histogram of price
            - Create a bar chart of city vs sales
            - What's the maximum value of score?
            - Show correlation between all numeric columns
            - Count the number of records by region
            - What is the 75th percentile of income?
            """)
        
        query = st.text_input("Enter your natural language query:", 
                             placeholder="e.g., What is the average age? or Plot a bar chart of department vs salary")
        
        if query:
            try:
                with st.spinner("🧠 Analyzing your query..."):
                    # Get LLM response
                    llm_output = get_llm_response(query, list(df.columns))
                    
                    # Clean and parse JSON
                    cleaned_output = clean_json(llm_output)
                    
                    # Debug info (optional, can be removed in production)
                    with st.expander("🔍 Debug: LLM Response"):
                        st.code(cleaned_output, language="json")
                    
                    try:
                        parsed = json.loads(cleaned_output)
                    except json.JSONDecodeError as e:
                        st.error(f"Failed to parse AI response: {str(e)}")
                        st.info("Please try rephrasing your query. Be specific about what you want to analyze.")
                        st.stop()
                    
                    # Extract parameters
                    operation = parsed.get("operation")
                    column = parsed.get("column")
                    groupby = parsed.get("groupby")
                    x_column = parsed.get("x_column")
                    y_column = parsed.get("y_column")
                    plot_type = parsed.get("plot_type")
                    value = parsed.get("value")
                    
                    # Resolve column names
                    resolved_column = resolve_column(column, list(df.columns)) if column else None
                    resolved_groupby = resolve_column(groupby, list(df.columns)) if groupby else None
                    resolved_x = resolve_column(x_column, list(df.columns)) if x_column else None
                    resolved_y = resolve_column(y_column, list(df.columns)) if y_column else None
                    
                    # Display interpretation
                    st.subheader("🎯 Interpretation")
                    st.markdown(f"**Operation:** `{operation}`")
                    if resolved_column:
                        st.markdown(f"**Column:** `{resolved_column}`")
                    if resolved_groupby:
                        st.markdown(f"**Group By:** `{resolved_groupby}`")
                    if resolved_x:
                        st.markdown(f"**X-Axis:** `{resolved_x}`")
                    if resolved_y:
                        st.markdown(f"**Y-Axis:** `{resolved_y}`")
                    if plot_type:
                        st.markdown(f"**Plot Type:** `{plot_type}`")
                    
                    # Execute based on operation type
                    st.subheader("📊 Result")
                    
                    if operation == "plot":
                        # Handle plotting
                        if resolved_x:
                            fig = create_plot(df, plot_type, resolved_x, resolved_y)
                            if fig:
                                st.pyplot(fig)
                                plt.close(fig)
                        else:
                            st.error("No column specified for plotting")
                    
                    elif operation == "correlation":
                        # Show correlation matrix
                        fig = create_plot(df, "correlation", None, None)
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                    
                    elif operation in ["mean", "sum", "max", "min", "count", "percentile", "iqr"]:
                        # Execute numeric operation
                        result, error = safe_execute_operation(df, operation, resolved_column, resolved_groupby, value)
                        
                        if error:
                            st.error(error)
                        else:
                            if isinstance(result, pd.Series):
                                st.success(f"**{operation.upper()} by {groupby}:**")
                                st.dataframe(result.reset_index(), use_container_width=True)
                            else:
                                st.success(f"**{operation.upper()} of {column}:** {result}")
                    
                    elif operation == "insight":
                        # Generate insights
                        with st.spinner("Generating insights..."):
                            insights = generate_insights(df)
                            st.markdown(insights)
                    
                    else:
                        st.warning(f"Operation '{operation}' is not yet supported. Try: mean, sum, max, min, count, plot, or correlation")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.error(traceback.format_exc())
                st.info("Please try rephrasing your query or check if the column names are correct.")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.info("Please make sure your CSV file is valid and not corrupted.")

else:
    # Show instructions when no file is uploaded
    st.info("👈 Please upload a CSV file to get started")
    
    with st.expander("📖 How to use DataSage AI"):
        st.markdown("""
        1. **Upload** your CSV dataset using the file uploader
        2. **Explore** the automatic data overview and statistics
        3. **Ask questions** in natural language like:
           - "What is the average age?"
           - "Show me total sales by region"
           - "Plot histogram of prices"
           - "What's the correlation between age and salary?"
        4. **Click** "Auto Analyze" for AI-generated insights
        
        **Tips:**
        - Be specific about column names
        - Use the exact column names as shown in the dataset
        - For plots, specify the chart type (bar, line, scatter, histogram)
        """)