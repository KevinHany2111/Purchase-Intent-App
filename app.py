import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# helpers must be imported, because the saved model uses add_features
from helpers import add_features, suggest_action

BASE_DIR = Path(__file__).parent

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

ACTION_TEXT = {
    "No action": "This visitor is likely to buy anyway. Do not show a discount.",
    "Show coupon": "This visitor is undecided. A coupon may help to close the sale.",
    "Show chat": "This visitor is unlikely to buy for now. Offer help with the chat assistant.",
}


@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "purchase_model.joblib")
    with open(BASE_DIR / "threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return model, threshold


st.set_page_config(page_title="Will This Visitor Buy?")
st.title("Will This Visitor Buy?")
st.write(
    "Enter the details of a browsing session. The app shows the chance of a "
    "purchase and suggests what the shop should do."
)

try:
    model, threshold = load_model()
except Exception:
    st.error(
        "The model files were not found. Run the notebook first to create "
        "purchase_model.joblib and threshold.json in the same folder as app.py."
    )
    st.stop()

with st.form("session_form"):
    st.subheader("Pages visited")
    col1, col2 = st.columns(2)
    admin_pages = col1.number_input("Account pages", min_value=0, value=0, step=1)
    admin_time = col2.number_input("Time on account pages (seconds)", min_value=0.0, value=0.0)
    info_pages = col1.number_input("Information pages", min_value=0, value=0, step=1)
    info_time = col2.number_input("Time on information pages (seconds)", min_value=0.0, value=0.0)
    product_pages = col1.number_input("Product pages", min_value=0, value=20, step=1)
    product_time = col2.number_input("Time on product pages (seconds)", min_value=0.0, value=600.0)

    st.subheader("Session behavior")
    col3, col4 = st.columns(2)
    bounce = col3.number_input(
        "Bounce rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.02, step=0.01, format="%.3f"
    )
    exit_rate = col4.number_input(
        "Exit rate (0 to 1)", min_value=0.0, max_value=1.0, value=0.04, step=0.01, format="%.3f"
    )
    page_value = col3.number_input(
        "Page value",
        min_value=0.0,
        value=0.0,
        help="Average value of the pages visited, calculated by the shop analytics. 0 means no valuable page was seen.",
    )
    special_day = col4.selectbox(
        "Closeness to a special day",
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        help="0 means far from a special day, 1 means very close.",
    )

    st.subheader("Visit details")
    col5, col6 = st.columns(2)
    month = col5.selectbox("Month", MONTHS, index=MONTHS.index("Nov"))
    visitor_type = col6.selectbox("Visitor type", ["Returning_Visitor", "New_Visitor", "Other"])
    weekend = st.checkbox("The visit is on a weekend")

    submitted = st.form_submit_button("Predict")

if submitted:
    # check the input before predicting
    problems = []
    pairs = [
        ("account", admin_pages, admin_time),
        ("information", info_pages, info_time),
        ("product", product_pages, product_time),
    ]
    for name, pages, seconds in pairs:
        if pages == 0 and seconds > 0:
            problems.append(f"You entered time on {name} pages but 0 {name} pages.")
    if admin_pages + info_pages + product_pages == 0:
        problems.append("Enter at least one visited page.")

    if problems:
        for p in problems:
            st.error(p)
    else:
        session = pd.DataFrame(
            [
                {
                    "Administrative": admin_pages,
                    "Administrative_Duration": admin_time,
                    "Informational": info_pages,
                    "Informational_Duration": info_time,
                    "ProductRelated": product_pages,
                    "ProductRelated_Duration": product_time,
                    "BounceRates": bounce,
                    "ExitRates": exit_rate,
                    "PageValues": page_value,
                    "SpecialDay": special_day,
                    "Month": month,
                    "VisitorType": visitor_type,
                    "Weekend": int(weekend),
                }
            ]
        )

        try:
            prob = float(model.predict_proba(session)[0, 1])
        except Exception:
            st.error("The prediction failed. Please check the values and try again.")
            st.stop()

        action = suggest_action(prob, threshold)

        st.subheader("Result")
        st.metric("Chance of a purchase", f"{prob:.0%}")
        st.progress(min(max(prob, 0.0), 1.0))
        st.write(f"**Suggested action: {action}**")
        st.write(ACTION_TEXT[action])

st.caption(
    "This tool is a decision aid. It is trained on sessions from one online shop "
    "and gives a score, not a guarantee."
)
