# app.py (modified to include an LLM panel)
import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from PIL import Image

# Optional LLM libs (conditionally imported)
USE_TRANSFORMERS = True  # toggle if you don't want to import transformers
try:
    if USE_TRANSFORMERS:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
except Exception:
    # we'll handle missing libs later
    pass

try:
    import openai
except Exception:
    openai = None

# ---------------------------
# Load data and train model
# ---------------------------
data = pd.read_excel('data1.xlsx')
y = data['label']
x = data.drop(['label'], axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# ---------------------------
# Prediction + stats functions
# ---------------------------
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = model.predict(input_data)
    return prediction[0]

def summary_statistics(crop):
    x = data[data['label'] == crop]
    stats = {
        'Minimum Nitrogen': x['N'].min(),
        'Average Nitrogen': x['N'].mean(),
        'Maximum Nitrogen': x['N'].max(),
        'Minimum Phosphorus': x['P'].min(),
        'Average Phosphorus': x['P'].mean(),
        'Maximum Phosphorus': x['P'].max(),
        'Minimum Potassium': x['K'].min(),
        'Average Potassium': x['K'].mean(),
        'Maximum Potassium': x['K'].max(),
        'Minimum Temperature': x['temperature'].min(),
        'Average Temperature': x['temperature'].mean(),
        'Maximum Temperature': x['temperature'].max(),
        'Minimum Humidity': x['humidity'].min(),
        'Average Humidity': x['humidity'].mean(),
        'Maximum Humidity': x['humidity'].max(),
        'Minimum pH': x['ph'].min(),
        'Average pH': x['ph'].mean(),
        'Maximum pH': x['ph'].max(),
        'Minimum Rainfall': x['rainfall'].min(),
        'Average Rainfall': x['rainfall'].mean(),
        'Maximum Rainfall': x['rainfall'].max()
    }
    return stats

def compare(conditions):
    avg_conditions = data[conditions].mean()
    st.write(f"Average Value for {conditions} is: {avg_conditions:.2f}")
    st.write("----------------------------------------------")
    for crop in data['label'].unique():
        avg_crop = data[data['label'] == crop][conditions].mean()
        st.write(f"{crop.capitalize()} : {avg_crop:.2f}")

# ---------------------------
# LLM helpers
# ---------------------------
@st.cache_resource
def load_local_model(model_name="gpt2"):
    """
    Load a local HF model for text-generation.
    Defaults to 'gpt2' (fast, small) — change to a larger model if you have GPU/VRAM.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if (os.environ.get("USE_GPU","0")=="1") else -1)
        return gen
    except Exception as e:
        st.error(f"Failed to load local model: {e}")
        return None

def generate_text_local(prompt, model_name="gpt2", max_length=200):
    gen = load_local_model(model_name)
    if gen is None:
        return "Local model not available. Check logs / dependencies."
    out = gen(prompt, max_length=max_length, do_sample=True, top_k=50, num_return_sequences=1)
    # pipeline returns list of dicts with 'generated_text'
    return out[0]['generated_text']

def generate_text_openai(prompt, model="gpt-3.5-turbo", max_tokens=256, temperature=0.7):
    """
    Generates text using OpenAI API. Requires OPENAI_API_KEY env var.
    """
    if openai is None:
        return "openai package not installed in environment."
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set in environment. Set it to use OpenAI."
    openai.api_key = api_key
    try:
        # Chat completion style
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role":"user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI call failed: {e}"

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(layout="wide")
st.title('AgroDataScience: Modernizing Agriculture with Data-driven Solutions')

# Styles
st.markdown("""
<style>
.big-font { font-size: 30px !important; font-weight: bold !important; color: #3366ff !important; }
.medium-font { font-size: 20px !important; color: #009933 !important; }
.error-msg { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Header images (if present)
cols = st.columns(3)
img_files = ['crop_header1.jpg', 'crop_header2.jpg', 'crop_header3.jpg']
for i, f in enumerate(img_files):
    try:
        with cols[i]:
            header_image = Image.open(f)
            st.image(header_image, use_column_width=True)
    except Exception:
        # skip missing images silently
        pass

st.markdown('<p class="big-font">Enter Climatic Conditions:</p>', unsafe_allow_html=True)

# Input fields with validation
N = st.text_input('Nitrogen (N) content (in kg/ha), range: 0-200: ')
P = st.text_input('Phosphorus (P) content (in kg/ha), range: 0-200: ')
K = st.text_input('Potassium (K) content (in kg/ha), range: 0-200: ')
temperature = st.text_input('Temperature (in Celsius), range: 0-50: ')
humidity = st.text_input('Humidity (in percentage), range: 0-100: ')
ph = st.text_input('pH level, range: 0-14: ')
rainfall = st.text_input('Rainfall (in mm), range: 0-1000: ')

error_msg = ""
# Range checks
def check_float_in_range(value, low, high):
    try:
        v = float(value)
        return low <= v <= high
    except Exception:
        return False

if N and not check_float_in_range(N,0,200): error_msg += "Please give Nitrogen input between the range 0 to 200.\n"
if P and not check_float_in_range(P,0,200): error_msg += "Please give Phosphorus input between the range 0 to 200.\n"
if K and not check_float_in_range(K,0,200): error_msg += "Please give Potassium input between the range 0 to 200.\n"
if temperature and not check_float_in_range(temperature,0,50): error_msg += "Please give Temperature input between the range 0 to 50.\n"
if humidity and not check_float_in_range(humidity,0,100): error_msg += "Please give Humidity input between the range 0 to 100.\n"
if ph and not check_float_in_range(ph,0,14): error_msg += "Please give pH input between the range 0 to 14.\n"
if rainfall and not check_float_in_range(rainfall,0,1000): error_msg += "Please give Rainfall input between the range 0 to 1000.\n"

if error_msg:
    st.markdown(f'<p class="error-msg">{error_msg}</p>', unsafe_allow_html=True)
else:
    # Prediction
    if st.button('Predict Crop'):
        suggested_crop = predict_crop(float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall))
        st.markdown(f'<p class="medium-font">The suggested crop for given climatic conditions is: <span style="color:green">{suggested_crop}</span></p>', unsafe_allow_html=True)

    # Summary stats
    st.markdown('<p class="big-font">Summary Statistics for Selected Crop:</p>', unsafe_allow_html=True)
    crop_selected = st.selectbox('Select Crop:', sorted(data['label'].unique()))
    if crop_selected:
        summary_stats = summary_statistics(crop_selected)
        stats_df = pd.DataFrame(summary_stats.items(), columns=['Statistic', 'Value'])
        st.table(stats_df)

    st.markdown('<p class="big-font">Comparison of Average Requirements for Each Crop with Average Conditions:</p>', unsafe_allow_html=True)
    conditions_selected = st.selectbox('Select Condition:', ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall'])
    if conditions_selected:
        compare(conditions_selected)

# ---------------------------
# LLM sidebar
# ---------------------------
st.sidebar.header("LLM Assistant")
provider = st.sidebar.selectbox("Provider", ["local (transformers)", "openai (cloud)"])

# provider-specific options
if provider.startswith("local"):
    st.sidebar.write("Local model: choose a Hugging Face model id (small models recommended if CPU only).")
    local_model_name = st.sidebar.text_input("Model name", value="gpt2")
    max_len = st.sidebar.slider("Max generation length", 50, 1024, 200)
else:
    st.sidebar.write("OpenAI provider: requires OPENAI_API_KEY in environment.")
    openai_model = st.sidebar.text_input("OpenAI model", value="gpt-3.5-turbo")
    temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

# Prompt area
st.sidebar.markdown("### Ask the LLM")
default_prompt = ("Given the following climatic and soil inputs, give a short explanation why the predicted crop "
                  "is suitable, and suggest 3 practical actions (fertilizer tips, planting time, irrigation) in bullet points.\n\n"
                  "Inputs:\n"
                  f"N={N}, P={P}, K={K}, temperature={temperature}, humidity={humidity}, ph={ph}, rainfall={rainfall}\n\n"
                  "Prediction (if available): " + (str(predict_crop(float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall))) if (N and P and K and temperature and humidity and ph and rainfall) else "N/A"))
prompt = st.sidebar.text_area("Prompt", value=default_prompt, height=220)

if st.sidebar.button("Ask LLM"):
    with st.spinner("Generating answer..."):
        if provider.startswith("local"):
            # local generation
            try:
                answer = generate_text_local(prompt, model_name=local_model_name, max_length=max_len)
            except Exception as e:
                answer = f"Local generation error: {e}"
        else:
            # openai
            answer = generate_text_openai(prompt, model=openai_model, max_tokens=256, temperature=temperature)
    st.sidebar.markdown("**LLM Answer:**")
    st.sidebar.write(answer)

# Footer images (optional)
footer_cols = st.columns(3)
footer_imgs = ['footer_image1.jpg', 'footer_image2.jpg', 'footer_image3.jpg']
for i, f in enumerate(footer_imgs):
    try:
        with footer_cols[i]:
            footer_image = Image.open(f)
            st.image(footer_image, use_column_width=True)
    except Exception:
        pass
