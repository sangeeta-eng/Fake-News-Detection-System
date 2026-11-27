
import streamlit as st
import joblib

# Load the trained model and vectorizer
model, vectorizer = joblib.load("fake_news_model.pkl")

# Website title
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰"
)

st.title("📰 Fake News Detection System")

st.write(
    "Enter a news headline or article "
    "to check the model's prediction."
)

# Input box
news_text = st.text_area(
    "Enter News Article",
    height=200,
    placeholder="Paste your news article here..."
)

# Prediction button
if st.button("Check News"):
    if news_text.strip() == "":
        st.warning("Please enter a news article.")
    else:
        # Convert text into numerical features
        text_features = vectorizer.transform([news_text])

        # Predict the news label
        prediction = model.predict(text_features)[0]

        # Display result
        if prediction == "FAKE":
            st.error("Prediction: FAKE NEWS")
        else:
            st.success("Prediction: REAL NEWS")

        st.info(
            "This is a Machine Learning prediction, "
            "not a guarantee of factual accuracy."
        )
