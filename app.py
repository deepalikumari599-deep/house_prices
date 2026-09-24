import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="DreamHome | House Price Predictor",
    page_icon="🏡",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #e8f8f5, #e8f4ff, #f5fbff);
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    color: #176b5b;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555;
    margin-bottom: 25px;
}

.card {
    background: rgba(255,255,255,0.90);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.result-box {
    background: linear-gradient(135deg, #176b5b, #2fa383);
    color: white;
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    margin-top: 20px;
}

.result-price {
    font-size: 38px;
    font-weight: 800;
}

.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 17px;
    font-weight: 700;
    background: #176b5b;
    color: white;
    border: none;
}

.stButton > button:hover {
    background: #0f5145;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    # IMPORTANT:
    # If your actual PKL filename is house_prices(2).pk1,
    # replace the filename below with that exact name.

    return joblib.load("house_prices.pk1")


model = load_model()


# =========================================================
# MODEL FEATURES
# =========================================================

FEATURES = list(model.feature_names_in_)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🏡 DreamHome</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">House Price Prediction System</div>',
    unsafe_allow_html=True
)


# =========================================================
# HOUSE ILLUSTRATION
# =========================================================

st.markdown("""
<div style="
    background:linear-gradient(#bde9ff,#eaf9ff);
    height:300px;
    border-radius:25px;
    position:relative;
    overflow:hidden;
    margin-bottom:30px;
">

    <!-- Sun -->
    <div style="
        position:absolute;
        right:80px;
        top:35px;
        width:65px;
        height:65px;
        background:#ffd45a;
        border-radius:50%;
    "></div>

    <!-- Ground -->
    <div style="
        position:absolute;
        bottom:0;
        left:0;
        width:100%;
        height:80px;
        background:#91cf7a;
    "></div>

    <!-- House -->
    <div style="
        position:absolute;
        bottom:55px;
        left:50%;
        transform:translateX(-50%);
        width:330px;
        height:145px;
        background:#fff5df;
        border:4px solid #8b6248;
        border-radius:4px;
    "></div>

    <!-- Roof -->
    <div style="
        position:absolute;
        bottom:200px;
        left:50%;
        transform:translateX(-50%);
        width:0;
        height:0;
        border-left:190px solid transparent;
        border-right:190px solid transparent;
        border-bottom:125px solid #b86f50;
    "></div>

    <!-- Left Window -->
    <div style="
        position:absolute;
        bottom:125px;
        left:calc(50% - 130px);
        width:65px;
        height:55px;
        background:#aee5f5;
        border:5px solid #6c503e;
        box-sizing:border-box;
    ">
        <div style="
            position:absolute;
            left:50%;
            top:0;
            width:4px;
            height:100%;
            background:#6c503e;
            transform:translateX(-50%);
        "></div>
        <div style="
            position:absolute;
            top:50%;
            left:0;
            width:100%;
            height:4px;
            background:#6c503e;
            transform:translateY(-50%);
        "></div>
    </div>

    <!-- Right Window -->
    <div style="
        position:absolute;
        bottom:125px;
        left:calc(50% + 65px);
        width:65px;
        height:55px;
        background:#aee5f5;
        border:5px solid #6c503e;
        box-sizing:border-box;
    ">
        <div style="
            position:absolute;
            left:50%;
            top:0;
            width:4px;
            height:100%;
            background:#6c503e;
            transform:translateX(-50%);
        "></div>
        <div style="
            position:absolute;
            top:50%;
            left:0;
            width:100%;
            height:4px;
            background:#6c503e;
            transform:translateY(-50%);
        "></div>
    </div>

    <!-- Door -->
    <div style="
        position:absolute;
        bottom:55px;
        left:50%;
        transform:translateX(-50%);
        width:65px;
        height:105px;
        background:#8b6248;
        border:4px solid #654735;
        border-bottom:none;
    ">
        <div style="
            position:absolute;
            right:8px;
            top:53px;
            width:8px;
            height:8px;
            background:#f5d77b;
            border-radius:50%;
        "></div>
    </div>

    <!-- Tree Left -->
    <div style="
        position:absolute;
        bottom:55px;
        left:40px;
        font-size:75px;
    ">🌳</div>

    <!-- Tree Right -->
    <div style="
        position:absolute;
        bottom:55px;
        right:40px;
        font-size:75px;
    ">🌳</div>

</div>
""", unsafe_allow_html=True)


# =========================================================
# INPUT SECTION
# =========================================================

st.markdown(
    '<div class="card">',
    unsafe_allow_html=True
)

st.subheader("🏠 Enter Property Details")

col1, col2, col3 = st.columns(3)


with col1:

    area = st.number_input(
        "Area",
        min_value=100.0,
        max_value=100000.0,
        value=1000.0,
        step=50.0
    )

    bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )


with col2:

    carpet_area = st.number_input(
        "Carpet Area",
        min_value=0.0,
        max_value=100000.0,
        value=800.0,
        step=50.0
    )

    bathroom = st.number_input(
        "Bathroom",
        min_value=1,
        max_value=20,
        value=2,
        step=1
    )


with col3:

    floor_number = st.number_input(
        "Floor Number",
        min_value=0,
        max_value=100,
        value=1,
        step=1
    )

    balcony = st.number_input(
        "Balcony",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )


# =========================================================
# LOCATION
# =========================================================

st.subheader("📍 Location")

location_features = [
    feature
    for feature in FEATURES
    if feature.lower().startswith("location_")
    or feature.lower().startswith("newlocation_")
]

locations = []

for feature in location_features:

    if "_" in feature:
        locations.append(feature.split("_", 1)[1])


if locations:

    location = st.selectbox(
        "Select Location",
        sorted(locations)
    )

else:

    location = st.text_input(
        "Enter Location"
    )


# =========================================================
# FURNISHING
# =========================================================

st.subheader("🛋️ Furnishing")

furnishing_options = [
    "Unfurnished",
    "Semi-Furnished",
    "Furnished"
]

furnishing = st.selectbox(
    "Furnishing Type",
    furnishing_options
)


st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

if st.button("🔮 Predict House Price"):

    try:

        # Create empty dataframe with EXACT model columns
        input_data = pd.DataFrame(
            0,
            index=[0],
            columns=FEATURES
        )

        # -------------------------------------------------
        # Numerical features
        # -------------------------------------------------

        numerical_values = {
            "Area": area,
            "BHK": bhk,
            "Carpet Area": carpet_area,
            "Bathroom": bathroom,
            "Balcony": balcony,
            "Floor_Number": floor_number
        }

        for column, value in numerical_values.items():

            if column in input_data.columns:
                input_data.loc[0, column] = value


        # -------------------------------------------------
        # Location
        # -------------------------------------------------

        possible_location_columns = [
            f"location_{location}",
            f"Location_{location}",
            f"newlocation_{location}",
            f"newlocation_{location.lower()}"
        ]

        for column in possible_location_columns:

            if column in input_data.columns:

                input_data.loc[0, column] = 1
                break


        # -------------------------------------------------
        # Furnishing
        # -------------------------------------------------

        furnishing_columns = [
            f"Furnishing_{furnishing}",
            f"furnishing_{furnishing}",
            f"Furnishing_{furnishing.replace('-', '-')}",
            f"furnishing_{furnishing.replace('-', '-')}"
        ]

        for column in furnishing_columns:

            if column in input_data.columns:

                input_data.loc[0, column] = 1
                break


        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = model.predict(input_data)[0]


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.markdown(
            f"""
            <div class="result-box">

                <div style="font-size:20px;">
                    🏡 Estimated House Price
                </div>

                <div class="result-price">
                    ₹ {prediction:,.2f}
                </div>

                <div style="margin-top:10px;">
                    Based on the property details provided
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


    except Exception as e:

        st.error("Prediction Error")

        st.code(str(e))


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <br>
    <div style="
        text-align:center;
        color:#777;
        font-size:14px;
    ">
        🏡 DreamHome House Price Predictor
    </div>
    """,
    unsafe_allow_html=True
)
