import pandas as pd
import streamlit as st
import html
from glob import glob
# Load and prepare data
@st.cache_data
def load_data():
    load_train = True
    if load_train:
        return pd.read_csv('train_dataset.csv')
    else:
        paths = glob('../.cache/jql-resources/**/results.jsonl', recursive=True)
        dfs = []
        for path in paths:
            df = pd.read_json(path, lines=True)
            df.rename(columns={'category': 'schema_name'}, inplace=True)
            dfs.append(df)
        return pd.concat(dfs)

# columns in the dataset
columns = ['title', 'abstract', 'url', 'schema_name', 'reasoning']

def make_clickable_links(url):
    """Convert URLs to clickable links with proper HTML escaping"""
    if pd.isna(url) or url == '' or str(url).strip() == '':
        return ''
    # Escape the URL to prevent HTML injection
    escaped_url = html.escape(str(url))
    return f'<a href="{escaped_url}" target="_blank">🔗 Link</a>'

def safe_html_escape(text):
    """Safely escape HTML characters in text"""
    if pd.isna(text):
        return ''
    return html.escape(str(text))

def main():
    st.title("Dataset Visualizer")
    st.write("Interactive dataset viewer with filtering capabilities")
    
    # Load data
    df = load_data()
    
    # Make URLs clickable using Streamlit's column configuration
    st.dataframe(
        df,
        column_config={
            "url": st.column_config.LinkColumn(
                "URL",
                help="Click to open the link",
                validate="^https?://.*",
                max_chars=100,
                display_text="🔗 Link"
            )
        },
        use_container_width=True
    )

if __name__ == "__main__":
    main()