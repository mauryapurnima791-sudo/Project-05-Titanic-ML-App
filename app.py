
import streamlit as st
import pandas as pd
import joblib

FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict the chance of survival.")

pclass = st.selectbox("Ticket Class", [1, 2, 3])
sex = st.radio("Sex", ["Female", "Male"])
age = st.slider("Age", 1, 80, 30)
fare = st.slider("Fare Paid", 0, 500, 30)

if st.button("Predict Survival"):
    try:
        model = load_model()

        passenger = pd.DataFrame(
            [[pclass, 1 if sex == "Female" else 0, age, fare]],
            columns=FEATURES
        )

        probability = model.predict_proba(passenger)[0][1]

        st.metric("Estimated Chance of Survival", f"{probability:.1%}")

        if probability >= 0.5:
            st.success("The model predicts: Likely to Survive")
        else:
            st.warning("The model predicts: Unlikely to Survive")

    except FileNotFoundError:
        st.error("Model file is missing. Please add model.joblib to this folder.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")

st.caption("Educational project using a Decision Tree trained on the Titanic dataset. This is not a real-world prediction.")
