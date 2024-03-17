import datetime
import tensorflow as tf
import streamlit as st
import tensorflow as tf
import pandas as pd
import altair as alt
#from utils import load_and_prep, preprocess_img, get_classes

class_names = ['apple_pie',
                'baby_back_ribs',
                'baklava',
                'beef_carpaccio',
                'beef_tartare',
                'beet_salad',
                'beignets',
                'bibimbap',
                'bread_pudding',
                'breakfast_burrito',
                'bruschetta',
                'caesar_salad',
                'cannoli',
                'caprese_salad',
                'carrot_cake',
                'ceviche',
                'cheesecake',
                'cheese_plate',
                'chicken_curry',
                'chicken_quesadilla',
                'chicken_wings',
                'chocolate_cake',
                'chocolate_mousse',
                'churros',
                'clam_chowder',
                'club_sandwich',
                'crab_cakes',
                'creme_brulee',
                'croque_madame',
                'cup_cakes',
                'deviled_eggs',
                'donuts',
                'dumplings',
                'edamame',
                'eggs_benedict',
                'escargots',
                'falafel',
                'filet_mignon',
                'fish_and_chips',
                'foie_gras',
                'french_fries',
                'french_onion_soup',
                'french_toast',
                'fried_calamari',
                'fried_rice',
                'frozen_yogurt',
                'garlic_bread',
                'gnocchi',
                'greek_salad',
                'grilled_cheese_sandwich',
                'grilled_salmon',
                'guacamole',
                'gyoza',
                'hamburger',
                'hot_and_sour_soup',
                'hot_dog',
                'huevos_rancheros',
                'hummus',
                'ice_cream',
                'lasagna',
                'lobster_bisque',
                'lobster_roll_sandwich',
                'macaroni_and_cheese',
                'macarons',
                'miso_soup',
                'mussels',
                'nachos',
                'omelette',
                'onion_rings',
                'oysters',
                'pad_thai',
                'paella',
                'pancakes',
                'panna_cotta',
                'peking_duck',
                'pho',
                'pizza',
                'pork_chop',
                'poutine',
                'prime_rib',
                'pulled_pork_sandwich',
                'ramen',
                'ravioli',
                'red_velvet_cake',
                'risotto',
                'samosa',
                'sashimi',
                'scallops',
                'seaweed_salad',
                'shrimp_and_grits',
                'spaghetti_bolognese',
                'spaghetti_carbonara',
                'spring_rolls',
                'steak',
                'strawberry_shortcake',
                'sushi',
                'tacos',
                'takoyaki',
                'tiramisu',
                'tuna_tartare',
                'waffles']

def get_classes():
    return class_names

def preprocess_img(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.uint8)
  return image

def load_and_prep(image, img_shape=224):
  image = preprocess_img(image)
  image = tf.image.resize(image, [img_shape, img_shape])
  return tf.cast(image, tf.float32)

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

st.set_page_config(page_title="Foodie",
                   page_icon="🍔")

#### SideBar ####

st.sidebar.title("What's Foodie ?")
st.sidebar.write("""
Our mission is unwavering – to remove the hassle from your daily culinary journey and bring joy back to your kitchen. With a simple photo, we unlock the potential of your ingredients and deliver a personalized list of recipes tailored to your preferences.

We created an easy to use and interactive application that allows your to take pictures of ingredients in the your fridge or cabinets and determines from those ingredients what recipes/meals you can make. This will provides real-time and instantaneous results and make cooking less of a hassle and more fun!

**Accuracy :** **`?`**
                 
**Model :** **`?`**

**Dataset :** **`?`**
""")


#### Main Body ####

st.title("Foodie🍔📷")
st.header("Identify recipes based on the foods you have!")
st.write("Add your food images below and select any allergies to generate a list of relevant recipes")
file = st.file_uploader(label="Upload a maximum of 5 items and a minimum of 1 item. Order the images you upload with the first image being the most important.",
                        type=["jpg", "jpeg", "png"], accept_multiple_files=True)
model = tf.keras.models.load_model("./models/07_model.hdf5")
option = st.selectbox(
    'Select from the dropdown list if any of these allergies apply to you',
    ('Peanuts', 'Tree Nuts', 'All Nuts','Milk','Eggs', 'Fish', 'Shellfish', 'Wheat', 'Soybeans', 'Not Applicable'))
st.write('You selected:', option)
st.sidebar.markdown("Created by **ADD**")
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