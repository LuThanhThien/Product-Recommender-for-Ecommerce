import pickle
import streamlit as st
import numpy as np
import pandas as pd
from SearchEngine import *

st.header("Search Engine")

input_path = 'inputs/data/amazon'
data_path = 'outputs/data/amazon'

# load the CSV file into a DataFrame
df = pd.read_csv(f'{data_path}\\amazon-product-web.csv')
searchEng = SearchEngine.TFIDFSearch()
searchEng.fit_transform(dataFrame=df, input_path=data_path)

# Create a search box for the user to enter their search query
top = 100
threshold = 0.20
user_query = st.text_input("What are you looking for?")
top_indices, top_simlarities = searchEng.search_query(user_query, top_number=top)

# Filter the product names based on the search query
top_products = searchEng.search_result(threshold=threshold)
corrected_query = searchEng.query
word_bank = searchEng.word_bank

# print(top_products)
print('User query: ', user_query)
print('Corrected query: ', corrected_query)
print(word_bank)
print(top_simlarities)

# website visual
max_image_size = 150
custom_css = f"""
    <style>
    .product-container {{
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }}
    .product-image {{
        max-width: {max_image_size}px;
        max-height: {max_image_size}px;
        margin-right: 20px;
    }}
    </style>
    """

# display products
if top_indices and user_query != "" and len(top_products) > 0:
    if user_query != corrected_query:
        st.markdown(f"*Search for \"{corrected_query}\" instead of \"{user_query}\".*")

    if len(top_products) < int(0.2*top):
        st.markdown(f"***Note:*** *The results may be less accurate for \"{corrected_query}\".*")
    else:
        st.write("Matching Products:")

    st.markdown(custom_css, unsafe_allow_html=True)
    for i, row in top_products.iterrows():
        # Display the image and text side by side
        st.markdown(
            f"""
            <div class="product-container">
                <img src="{row['img_link']}" alt="{row['product_name']}" class="product-image">
                <div>
                    <p>{row['product_name']}</p>
                    <a href="{row['product_link']}" target="_blank">See Product</a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

elif user_query != "" and len(top_products) == 0:
    st.write(f"No matching products found for \"{user_query}\".")
