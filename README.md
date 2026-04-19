# 🧠 DataSage AI

**AI-Powered Natural Language Data Analysis Platform**

DataSage AI is an intelligent data analysis assistant that converts natural language queries into executable data operations, making data analysis accessible to everyone regardless of technical expertise.

The system leverages **LLM-powered query understanding**, **smart visualization generation**, and **automated insight extraction** to provide instant, accurate data analysis.

---

## 🌐 Live Demo

> 🔗 https://datasage-ai-iota.streamlit.app/

---

## ✨ Key Features

### 🗣️ Natural Language to Data Operations  
Users type plain English questions, and DataSage AI automatically converts them into structured data operations. No coding or SQL required.

**Examples:**
- "What is the average age?" → `mean` operation
- "Show total sales by region" → `sum` with `groupby`
- "Plot histogram of prices" → `histogram` visualization

---

### 🎯 Intelligent Query Parsing  
Using Groq's Llama 3.3 70B model, the system understands:

- Statistical operations (mean, sum, max, min, count)
- Grouped aggregations
- Visualization requests
- Correlation analysis
- Percentile and IQR calculations

---

### 📊 Smart Visualizations  
Automatically generates appropriate charts based on query context:

| Plot Type | Use Case |
|-----------|----------|
| Bar Chart | Categorical comparisons |
| Line Plot | Trends over time |
| Scatter Plot | Variable relationships |
| Histogram | Distribution analysis |
| Box Plot | Outlier detection |
| Heatmap | Correlation matrices |

---

### 🔍 Fuzzy Column Matching  
The system intelligently matches column names even when users don't remember exact names:

- Case-insensitive matching
- Close-match suggestions (typo tolerance)
- Automatic column resolution

*"Show me total revnue by regon"* → Correctly maps to "revenue" and "region"

---

### 🤖 AI-Generated Insights  
One-click generation of 5-7 meaningful insights including:

- Patterns and trends
- Relationships between variables
- Anomaly detection
- Statistical summaries

---

### 📈 Real-Time Analysis  
Instant execution of queries with:

- Live data preview
- Dynamic result rendering
- Interactive visualizations
- Downloadable outputs

---

### 🎨 Clean, Professional UI  
Built with Streamlit featuring:

- Automatic data profiling
- Expandable sections
- Dark/light mode support
- Mobile-responsive design

---

## 🖥 System Architecture
