import streamlit as st
import pandas as pd
import numpy as np
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import io

# Import required libraries
try:
    from anthropic import Anthropic
    from pydantic import BaseModel, Field
    from docx import Document
    from docx.shared import Inches
except ImportError as e:
    st.error(f"Missing dependency: {e}. Please install with: uv add anthropic pydantic python-docx")
    st.stop()

# Pydantic models for structured outputs
class Anomaly(BaseModel):
    metric: str = Field(description="The financial metric with anomaly")
    current_value: str = Field(description="Current quarter value")
    comparison_value: str = Field(default="N/A", description="Previous quarter value") 
    change_percent: str = Field(default="N/A", description="Percentage change")
    risk_level: str = Field(description="High, Medium, or Low")
    explanation: str = Field(description="Business context explanation", max_length=200)
    next_steps: str = Field(description="Recommended actions", max_length=150)
    z_score: float = Field(description="Statistical z-score")

class ExecutiveSummary(BaseModel):
    narrative: str = Field(description="3-5 sentence executive summary")
    key_metrics: List[str] = Field(description="List of key metrics mentioned")
    data_sources: List[str] = Field(description="Source columns referenced")

class FinancialAnalysis(BaseModel):
    executive_summary: ExecutiveSummary
    anomalies: List[Anomaly]
    total_anomalies_found: int
    analysis_timestamp: str

# Initialize Anthropic client
def get_anthropic_client():
    """Get Anthropic client with API key validation"""
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except:
        api_key = None
    
    if not api_key:
        api_key = st.sidebar.text_input(
            "Anthropic API Key", type="password", 
            help="Get your API key from console.anthropic.com"
        )
    
    if not api_key:
        st.warning("Please enter your Anthropic API key in the sidebar to continue")
        st.stop()
    
    return Anthropic(api_key=api_key)

# Database setup for audit trail
def init_database():
    conn = sqlite3.connect('audit_trail.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS analysis_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            input_file_name TEXT,
            anomalies_detected INTEGER,
            prompt_used TEXT,
            ai_response TEXT,
            user_edits TEXT
        )
    ''')
    conn.commit()
    return conn

# Anomaly detection functions
def detect_anomalies(df: pd.DataFrame, threshold: float = 2.0) -> Dict[str, Any]:
    """Detect statistical anomalies using z-score and IQR methods"""
    anomalies = []
    
    # Get numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if len(df[col].dropna()) < 2:  # Skip if insufficient data
            continue
            
        # Calculate z-scores
        mean_val = df[col].mean()
        std_val = df[col].std()
        
        if std_val == 0:  # Skip if no variation
            continue
            
        z_scores = np.abs((df[col] - mean_val) / std_val)
        
        # Find outliers
        outlier_indices = z_scores >= threshold
        
        if outlier_indices.any():
            for idx in df[outlier_indices].index:
                anomalies.append({
                    'metric': col,
                    'index': idx,
                    'value': df.loc[idx, col],
                    'z_score': z_scores.loc[idx],
                    'mean': mean_val,
                    'std': std_val
                })
    
    return anomalies

def calculate_qoq_changes(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate quarter-over-quarter changes"""
    if 'Quarter' not in df.columns:
        return df
        
    df_sorted = df.sort_values('Quarter')
    numeric_cols = df_sorted.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        df_sorted[f'{col}_QoQ_Change'] = df_sorted[col].pct_change() * 100
        df_sorted[f'{col}_QoQ_Abs_Change'] = df_sorted[col].diff()
    
    return df_sorted

# LLM Analysis
async def analyze_with_claude(df: pd.DataFrame, anomalies: List[Dict], client: Anthropic) -> FinancialAnalysis:
    """Generate analysis using Claude Haiku with function calling"""
    
    # Prepare data summary
    data_summary = {
        'columns': df.columns.tolist(),
        'shape': df.shape,
        'numeric_summary': df.describe().to_dict(),
        'latest_quarter': df.iloc[-1].to_dict() if len(df) > 0 else {},
        'anomalies_detected': len(anomalies)
    }
    
    # Create the analysis tool
    analysis_tool = {
        "name": "financial_analysis",
        "description": "Analyze quarterly financial data and generate executive summary with anomaly explanations",
        "input_schema": {
            "type": "object",
            "properties": {
                "executive_summary": {
                    "type": "object",
                    "properties": {
                        "narrative": {"type": "string", "description": "3-5 sentence executive summary"},
                        "key_metrics": {"type": "array", "items": {"type": "string"}},
                        "data_sources": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["narrative", "key_metrics", "data_sources"]
                },
                "anomalies": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "metric": {"type": "string"},
                            "current_value": {"type": "string"},
                            "comparison_value": {"type": "string"},
                            "change_percent": {"type": "string"},
                            "risk_level": {"type": "string", "enum": ["High", "Medium", "Low"]},
                            "explanation": {"type": "string", "maxLength": 80},
                            "next_steps": {"type": "string"},
                            "z_score": {"type": "number"}
                        },
                                                                "required": ["metric", "current_value", "risk_level", "explanation", "next_steps", "z_score"]
                    }
                },
                "total_anomalies_found": {"type": "integer"},
                "analysis_timestamp": {"type": "string"}
            },
            "required": ["executive_summary", "anomalies", "total_anomalies_found", "analysis_timestamp"]
        }
    }
    
    # Construct prompt
    prompt = f"""You are analyzing quarterly financial data for executive reporting.

Dataset Overview:
- Columns: {data_summary['columns']}
- Shape: {data_summary['shape']} (rows, columns)
- Latest Quarter Data: {json.dumps(data_summary['latest_quarter'], indent=2)}

Statistical Anomalies Detected: {len(anomalies)}
{json.dumps(anomalies, indent=2) if anomalies else "None"}

Instructions:
1. Generate a 3-5 sentence executive summary highlighting key trends
2. For each anomaly, provide business context and next steps  
3. Use ONLY the numbers provided - no estimates or assumptions
4. Focus on actionable insights for compliance managers
5. Keep explanations under 80 words each

Use the financial_analysis tool to structure your response."""

    try:
        response = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=2000,
            tools=[analysis_tool],
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract tool use result
        tool_result = None
        for content in response.content:
            if content.type == "tool_use":
                tool_result = content.input
                break
        
        if not tool_result:
            st.error("Claude did not use the analysis tool properly")
            return None
            
        # Convert to Pydantic model for validation
        return FinancialAnalysis(**tool_result)
        
    except Exception as e:
        st.error(f"Error calling Claude API: {str(e)}")
        return None

# Export functions
def export_to_word(analysis: FinancialAnalysis, filename: str = "financial_analysis.docx"):
    """Export analysis to Word document"""
    doc = Document()
    
    # Title
    title = doc.add_heading('Quarterly Financial Analysis', 0)
    
    # Executive Summary
    doc.add_heading('Executive Summary', level=1)
    doc.add_paragraph(analysis.executive_summary.narrative)
    
    # Key Metrics
    doc.add_heading('Key Metrics', level=2)
    for metric in analysis.executive_summary.key_metrics:
        doc.add_paragraph(f"• {metric}", style='List Bullet')
    
    # Anomalies Section
    if analysis.anomalies:
        doc.add_heading('Risk & Anomaly Analysis', level=1)
        
        for i, anomaly in enumerate(analysis.anomalies, 1):
            doc.add_heading(f'Anomaly {i}: {anomaly.metric}', level=2)
            doc.add_paragraph(f"Current Value: {anomaly.current_value}")
            if anomaly.comparison_value:
                doc.add_paragraph(f"Previous Value: {anomaly.comparison_value}")
            if anomaly.change_percent:
                doc.add_paragraph(f"Change: {anomaly.change_percent}")
            doc.add_paragraph(f"Risk Level: {anomaly.risk_level}")
            doc.add_paragraph(f"Analysis: {anomaly.explanation}")
            doc.add_paragraph(f"Recommended Actions: {anomaly.next_steps}")
            doc.add_paragraph(f"Statistical Confidence: z-score = {anomaly.z_score:.2f}")
    
    # Audit Information
    doc.add_heading('Audit Trail', level=1)
    doc.add_paragraph(f"Analysis Generated: {analysis.analysis_timestamp}")
    doc.add_paragraph(f"Total Anomalies Detected: {analysis.total_anomalies_found}")
    doc.add_paragraph("Data Sources: " + ", ".join(analysis.executive_summary.data_sources))
    
    # Save to bytes
    doc_bytes = io.BytesIO()
    doc.save(doc_bytes)
    doc_bytes.seek(0)
    
    return doc_bytes.getvalue()

# Streamlit App
def main():
    st.set_page_config(
        page_title="AI Anomaly Detection and Summarization for Reporting",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🚀 AI Anomaly Detection and Summarization for Reporting")
    st.markdown("*Automatically generate executive summaries and detect statistical anomalies from financial data*")
    
    # Initialize database
    conn = init_database()
    client = get_anthropic_client()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    z_score_threshold = st.sidebar.slider("Anomaly Detection Threshold (Z-Score)", 1.5, 3.0, 2.0, 0.1)
    
    # File upload
    st.header("📁 Upload Financial Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV or Excel file", 
        type=['csv', 'xlsx'],
        help="Upload quarterly financial data for analysis"
    )
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ File uploaded: {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} columns)")
            
            # Display data preview
            st.header("📊 Data Preview")
            st.dataframe(df.head())
            
            # Calculate QoQ changes
            df_with_changes = calculate_qoq_changes(df)
            
            # Detect anomalies
            with st.spinner("🔍 Detecting anomalies..."):
                anomalies = detect_anomalies(df_with_changes, threshold=z_score_threshold)
            
            st.header("⚠️ Anomalies Detected")
            if anomalies:
                st.warning(f"Found {len(anomalies)} statistical anomalies")
                
                for anomaly in anomalies:
                    with st.expander(f"🚨 {anomaly['metric']} (z-score: {anomaly['z_score']:.2f})"):
                        st.write(f"**Value:** {anomaly['value']:,.2f}")
                        st.write(f"**Dataset Mean:** {anomaly['mean']:,.2f}")
                        st.write(f"**Standard Deviation:** {anomaly['std']:,.2f}")
            else:
                st.info("No significant anomalies detected")
            
            # Generate AI analysis
            if st.button("🤖 Generate AI Analysis", type="primary"):
                with st.spinner("🧠 Analyzing with Claude Haiku..."):
                    
                    # Note: Using sync version for Streamlit compatibility
                    # In a real async environment, you'd use the async version
                    analysis = None
                    
                    try:
                        # Prepare data for Claude
                        data_summary = {
                            'columns': df.columns.tolist(),
                            'shape': df.shape,
                            'numeric_summary': df_with_changes.describe().to_dict(),
                            'latest_quarter': df_with_changes.iloc[-1].to_dict() if len(df_with_changes) > 0 else {},
                        }
                        
                        # Simplified synchronous version
                        analysis_tool = {
                            "name": "financial_analysis",
                            "description": "Analyze financial data and generate structured insights",
                            "input_schema": {
                                "type": "object",
                                "properties": {
                                    "executive_summary": {
                                        "type": "object",
                                        "properties": {
                                            "narrative": {"type": "string"},
                                            "key_metrics": {"type": "array", "items": {"type": "string"}},
                                            "data_sources": {"type": "array", "items": {"type": "string"}}
                                        }
                                    },
                                    "anomalies": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "metric": {"type": "string"},
                                                "current_value": {"type": "string"},
                                                "comparison_value": {"type": "string"},
                                                "change_percent": {"type": "string"},
                                                "risk_level": {"type": "string"},
                                                "explanation": {"type": "string"},
                                                "next_steps": {"type": "string"},
                                                "z_score": {"type": "number"}
                                            }
                                        }
                                    },
                                    "total_anomalies_found": {"type": "integer"},
                                    "analysis_timestamp": {"type": "string"}
                                }
                            }
                        }
                        
                        prompt = f"""Analyze this quarterly financial data and provide insights:

Data: {json.dumps(data_summary['latest_quarter'], indent=2)}
Statistical Anomalies: {json.dumps(anomalies, indent=2)}

Requirements:
1. Generate executive summary (3-5 sentences max)
2. For each anomaly, provide:
   - Business explanation (max 200 characters)
   - Next steps (max 150 characters)
3. Use only provided numbers - no estimates
4. Keep language concise and professional

CRITICAL: Keep all explanations and next steps very brief - under the character limits."""

                        response = client.messages.create(
                            model="claude-3-5-haiku-20241022",
                            max_tokens=2000,
                            tools=[analysis_tool],
                            messages=[{"role": "user", "content": prompt}]
                        )
                        
                        # Parse response
                        tool_result = None
                        for content in response.content:
                            if content.type == "tool_use":
                                tool_result = content.input
                                break
                        
                        if tool_result:
                            # Add timestamp
                            tool_result["analysis_timestamp"] = datetime.now().isoformat()
                            analysis = FinancialAnalysis(**tool_result)
                    
                    except Exception as e:
                        st.error(f"Error generating analysis: {str(e)}")
                        st.info("Falling back to basic analysis...")
                        
                        # Fallback analysis
                        analysis = FinancialAnalysis(
                            executive_summary=ExecutiveSummary(
                                narrative="Analysis completed successfully. Review the detected anomalies for potential risks.",
                                key_metrics=[col for col in df.columns if df[col].dtype in ['int64', 'float64']][:5],
                                data_sources=df.columns.tolist()[:5]
                            ),
                            anomalies=[
                                Anomaly(
                                    metric=anomaly['metric'],
                                    current_value=f"{anomaly['value']:,.2f}",
                                    comparison_value="N/A",
                                    change_percent="N/A", 
                                    risk_level="Medium" if anomaly['z_score'] < 3 else "High",
                                    explanation=f"Statistical outlier detected with z-score of {anomaly['z_score']:.2f}",
                                    next_steps="Review data source and validate business context",
                                    z_score=anomaly['z_score']
                                ) for anomaly in anomalies[:5]  # Limit to 5 anomalies
                            ],
                            total_anomalies_found=len(anomalies),
                            analysis_timestamp=datetime.now().isoformat()
                        )
                
                if analysis:
                    # Display results
                    st.header("📝 AI-Generated Analysis")
                    
                    # Executive Summary
                    st.subheader("Executive Summary")
                    st.write(analysis.executive_summary.narrative)
                    
                    # Key Metrics
                    with st.expander("Key Metrics Referenced"):
                        for metric in analysis.executive_summary.key_metrics:
                            st.write(f"• {metric}")
                    
                    # Anomaly Analysis
                    if analysis.anomalies:
                        st.subheader("🚨 Risk & Anomaly Analysis")
                        
                        for i, anomaly in enumerate(analysis.anomalies, 1):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.write(f"**{i}. {anomaly.metric}**")
                                    st.write(f"*Current:* {anomaly.current_value}")
                                    if anomaly.comparison_value and anomaly.comparison_value != "N/A":
                                        st.write(f"*Previous:* {anomaly.comparison_value}")
                                    if anomaly.change_percent and anomaly.change_percent != "N/A":
                                        st.write(f"*Change:* {anomaly.change_percent}")
                                    st.write(f"*Analysis:* {anomaly.explanation}")
                                    st.write(f"*Next Steps:* {anomaly.next_steps}")
                                
                                with col2:
                                    risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                                    st.metric(
                                        "Risk Level", 
                                        f"{risk_color.get(anomaly.risk_level, '⚪')} {anomaly.risk_level}",
                                        f"z-score: {anomaly.z_score:.2f}"
                                    )
                                
                                st.divider()
                    
                    # Export options
                    st.header("📤 Export Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("📄 Download Word Report"):
                            doc_bytes = export_to_word(analysis)
                            st.download_button(
                                label="💾 Download DOCX",
                                data=doc_bytes,
                                file_name="financial_analysis_report.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )
                    
                    with col2:
                        json_data = analysis.model_dump_json(indent=2)
                        st.download_button(
                            label="📊 Download JSON",
                            data=json_data,
                            file_name="financial_analysis.json",
                            mime="application/json"
                        )
                    
                    # Audit trail
                    conn.execute('''
                        INSERT INTO analysis_log 
                        (timestamp, input_file_name, anomalies_detected, prompt_used, ai_response) 
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        datetime.now().isoformat(),
                        uploaded_file.name,
                        len(anomalies),
                        "Financial analysis prompt",
                        json.dumps(analysis.model_dump(), indent=2)
                    ))
                    conn.commit()
                    
                    st.success("✅ Analysis complete! Audit trail logged.")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("*AI-powered anomaly detection and summarization for enterprise reporting*")

if __name__ == "__main__":
    main()