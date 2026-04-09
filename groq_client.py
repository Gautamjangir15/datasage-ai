from groq import Groq 
import json
import re
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def get_llm_response(query, columns):
    prompt = f"""
    You are a data analyst agent. Convert the user query into a JSON plan.

    Available columns: {columns}

    Return JSON with this EXACT structure:
    {{
        "operation": "mean|sum|max|min|count|plot|correlation|insight",
        "column": "column_name",
        "groupby": "optional_column_name",
        "x_column": "column_name_for_x_axis",
        "y_column": "column_name_for_y_axis",
        "plot_type": "bar|line|scatter|histogram|box",
        "value": null,
        "steps": [
            {{
                "operation": "groupby|sort|limit",
                "column": "column_name",
                "groupby": "group_column",
                "order": "asc|desc",
                "value": 5
            }}
        ]
    }}

    Rules:
    - For simple aggregations (mean, sum, etc.), use "operation" and "column"
    - For grouped operations, also include "groupby"
    - For plots, use "plot" operation with "x_column", "y_column", and "plot_type"
    - For "sort" steps, include "order"
    - For "limit" steps, include "value"
    - Use exact column names from the list above
    - Output ONLY valid JSON, no other text

    User Query: {query}
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1  # Lower temperature for more consistent JSON
    )
    
    return response.choices[0].message.content


def clean_json(response):
    # Remove markdown code blocks
    response = re.sub(r"```json|```", "", response)

    # Extract JSON object
    match = re.search(r"\{.*\}", response, re.DOTALL)

    return match.group(0).strip() if match else response.strip()


def generate_insights(df, query=None):
    summary = df.describe().to_string()
    correlation = df.corr(numeric_only=True).to_string()
    missing = df.isnull().sum().to_string()

    prompt = f"""
    You are a senior data analyst.

    Analyze the dataset and provide key insights.

    DATA SUMMARY:
    {summary}

    CORRELATION:
    {correlation}

    MISSING VALUES:
    {missing}

    Instructions:
    - Give 5-7 meaningful insights
    - Focus on patterns, relationships, anomalies
    - Avoid just repeating numbers
    - Be concise and clear
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content