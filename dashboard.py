import streamlit as st
import pandas as pd
import requests
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import os 

# GitHub Access token 설정
GITHUB_API_URL = "https://api.github.com/repos"
OWNER = "KevinTheRainmaker"  # 깃허브 사용자명
REPO = "B-Peach-Evaluation"  # 레포지토리 이름
FOLDER_PATH = "results"  # 폴더 경로

# Load API key from environment variables or .env file
if os.environ.get("GITHUB_ACTIONS") is None:
    from dotenv import load_dotenv
    print('Loading API Key from local .env file...')
    load_dotenv()
    ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')
else:
    print('Loading API Key from Streamlit Secrets...')
    ACCESS_TOKEN = st.secrets["ACCESS_TOKEN"]

# Get the list of files in the GitHub folder
# @st.cache_data
def get_file_list():
    headers = {"Authorization": f"token {ACCESS_TOKEN}"}
    url = f"{GITHUB_API_URL}/{OWNER}/{REPO}/contents/{FOLDER_PATH}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        files = response.json()
        return [file['name'] for file in files if file['name'].endswith('.csv')]
    else:
        st.error("❌ Failed to fetch files from GitHub.")
        return []

# Fetch the content of a specific CSV file
# @st.cache_data
def fetch_csv_content(file_name):
    headers = {"Authorization": f"token {ACCESS_TOKEN}"}
    url = f"{GITHUB_API_URL}/{OWNER}/{REPO}/contents/{FOLDER_PATH}/{file_name}"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        file_content = response.json()['content']
        decoded_content = StringIO(base64.b64decode(file_content).decode('utf-8'))
        return pd.read_csv(decoded_content)
    else:
        st.error(f"❌ Failed to fetch the file: {file_name}")
        return pd.DataFrame()

# Aggregate all CSV files into one DataFrame
def aggregate_csv_files(file_names):
    all_data = []
    for file_name in file_names:
        df = fetch_csv_content(file_name)
        if not df.empty:
            df['span_count'] = df['original_passage'].str.count(r'<span')  # Calculate <span> tag count
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

# Streamlit UI
st.title("🍑 B-Peach EM Score Dashboard")

files = get_file_list()
if files:
    st.sidebar.write(f"📂 Found {len(files)} CSV files in the folder.")
    
    # Use session state to store selected file
    if "selected_file" not in st.session_state:
        st.session_state.selected_file = files[0]  # Default to first file

    selected_file = st.sidebar.selectbox(
        "Select a file to preview",
        files,
        index=files.index(st.session_state.selected_file),
        key="selected_file"
    )

    # Aggregate all data
    with st.spinner("📊 Aggregating all CSV data..."):
        aggregated_data = aggregate_csv_files(files)

    if not aggregated_data.empty:
        st.subheader("📋 Aggregated Data Overview")
        st.dataframe(aggregated_data.head(10))

        # Load the selected file separately for statistics
        with st.spinner(f"📄 Loading {selected_file} statistics..."):
            selected_data = fetch_csv_content(selected_file)
            if not selected_data.empty:
                selected_data['span_count'] = selected_data['original_passage'].str.count(r'<span')

                # Display statistics for the selected file
                st.sidebar.subheader(f"📊 Statistics for {selected_file}")
                st.sidebar.write(selected_data.describe())
        
        # Visualize with a boxplot
        st.subheader("📈 Boxplot of EM Score by Number of <span> Tags")
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='span_count', y='em_score', data=aggregated_data, palette="coolwarm")
        plt.title('Distribution of EM Scores by Number of <span> Tags', fontsize=16, fontweight='bold')
        plt.xlabel('Number of <span> Tags', fontsize=12)
        plt.ylabel('EM Score', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        st.pyplot(plt)
    else:
        st.warning("⚠️ No data to visualize.")
else:
    st.warning("⚠️ No CSV files found in the specified folder.")
