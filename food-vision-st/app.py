import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
from utils import load_and_prep, preprocess_img, get_classes

#@st.cache_resource(suppress_st_warning=True)#
def predicting(image, model):
    image = load_and_prep(image)
    predictions = model.predict(tf.expand_dims(image, axis=0))
    pred_class = class_names[tf.argmax(predictions, axis=1)[0].numpy()]
    pred_conf = tf.reduce_max(predictions[0])
    top_5 = sorted((predictions.argsort())[0][-5:][::-1])
    values = predictions[0][top_5] * 100
    labels = []
    for x in range(5):
        labels.append(class_names[top_5[x]])
    df = pd.DataFrame({"Top 5 Predictions": labels,
                       "F1 Scores": values,
                       'color': ['#EC5953', '#EC5953', '#EC5953', '#EC5953', '#EC5953']})
    df = df.sort_values('F1 Scores')
    return pred_class, pred_conf, df

class_names = get_classes()

st.set_page_config(page_title="Food Vision",
                   page_icon="🍔")

#### SideBar ####

st.sidebar.title("What's Food Vision ?")
st.sidebar.write("""
FoodVision is an end-to-end **CNN Image Classification Model** which identifies the food in your image. 

It can identify over 100 different food classes

It is based upom a pre-trained Image Classification Model that comes with Keras and then retrained on the infamous **Food101 Dataset**.

**Accuracy :** **`80% +`**

**Model :** **`EfficientNetB0`**

**Dataset :** **`Food101`**
""")


#### Main Body ####

st.title("Food Vision 🍔📷")
st.header("Identify what's in your food photos!")
st.write("To know more about this app, visit [**GitHub**](https://github.com/boradj/food-vision)")
file = st.file_uploader(label="Upload an image of food.",
                        type=["jpg", "jpeg", "png"])


model = tf.keras.models.load_model("/content/07_efficientnetb0_fine_tuned_101_classes_mixed_precision_80_validation.h5")


st.sidebar.markdown("Created by **Jaydip Borad**")
st.sidebar.markdown(body="""

<th style="border:None"><a href="https://twitter.com/jdborad" target="blank"><img align="center" src="https://bit.ly/3wK17I6" alt="jdborad" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://www.linkedin.com/in/jaydip-borad/" target="blank"><img align="center" src="https://bit.ly/3wCl82U" alt="jaydipborad" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://github.com/boradj" target="blank"><img align="center" src="https://bit.ly/githubjd" alt="boradj" height="40" width="40" /></a></th>
<th style="border:None"><a href="https://www.instagram.com/borad.j/" target="blank"><img align="center" src="https://bit.ly/3oZABHZ" alt="borad.j" height="40" width="40" /></a></th>

""", unsafe_allow_html=True)

if not file:
    st.warning("Please upload an image")
    st.stop()

else:
    image = file.read()
    st.image(image, use_column_width=True)
    pred_button = st.button("Predict")

if pred_button:
    pred_class, pred_conf, df = predicting(image, model)
    st.success(f'Prediction : {pred_class} \nConfidence : {pred_conf*100:.2f}%')
    st.write(alt.Chart(df).mark_bar().encode(
        x='F1 Scores',
        y=alt.X('Top 5 Predictions', sort=None),
        color=alt.Color("color", scale=None),
        text='F1 Scores'
    ).properties(width=600, height=400))
