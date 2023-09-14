import pickle
import streamlit as st
import numpy as np
import pandas as pd
from SearchEngine import *

st.header("ThreeRrific")

input_path = 'inputs/data/amazon'
data_path = 'outputs/data/amazon'

# load the CSV file into a DataFrame
df = pd.read_csv(f'{data_path}/amazon-product-web.csv')
searchEng = TFIDFSearch()
searchEng.fit_transform(dataFrame=df, input_path=data_path)


# Create a search box for the user to enter their search query
top = 50
threshold = 0.10
input_query = st.text_input("What are you looking for?")
user_query = input_query.lower()
top_indices, top_simlarities = searchEng.search_query(user_query, top_number=top)

# Filter the product names based on the search query
top_products = searchEng.search_result(threshold=threshold)
corrected_query = searchEng.query
word_bank = searchEng.word_bank

# print(top_products)
# print('User query: ', user_query)
# print('Corrected query: ', corrected_query)
# print(word_bank)
# print(top_simlarities)

# website visual
default_image_size = 150
custom_css = f"""
    <style>
    .product-container {{
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }}
    .product-image {{
        width: {default_image_size}px;
        height: {default_image_size}px;
        object-fit: contain;
        margin-right: 20px;
    }}
    </style>
    """

# display products
if top_indices and user_query != "" and len(top_products) > 0:

    if user_query != corrected_query:
        st.markdown(f"*Search for \"{corrected_query}\" instead of \"{user_query}\".*")

    if len(top_products) < int(0.2*top):
        st.markdown(f"*The results may be less accurate for \"{corrected_query}\".*")
    # else:
    #     st.write("Matching Products:")

    # Add custom CSS for styling
    custom_css = f"""
                <style>
                .product-container {{
                    display: flex;
                    align-items: center;
                    margin-bottom: 20px;
                    padding: 10px;
                    border: 2px solid #636262;
                    transition: transform 0.2s, border-color 0.2s, box-shadow 0.2s;
                }}
                .product-container:hover {{
                    transform: scale(1.05);  /* Zoom effect on hover */
                    border-color: #ff5722;  /* Border color on hover */
                    box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);  /* Box shadow on hover */
                }}
                .product-image {{
                    width: {default_image_size}px;
                    height: {default_image_size}px; 
                    margin-right: 20px;
                    object-fit: contain;
                }}
                .product-details {{
                    flex: 1;
                }}
                </style>
                """

    st.markdown(custom_css, unsafe_allow_html=True)
    for i, row in top_products.iterrows():
        # Limit the product name to 300 characters
        product_name = row['product_name'][:300] + ('...' if len(row['product_name']) > 300 else '')

        # Display the image, text, and link inside an anchor element
        st.markdown(
            f"""
            <a href="{row['product_link']}" target="_blank" style="text-decoration: none; color: inherit;">
                <div class="product-container">
                    <img src="{row['img_link']}" alt="{product_name}" class="product-image">
                    <div class="product-details">
                        <p class="product-name"; font-size: 17px;>{product_name}</p>
                        <p style="font-size: 22px; color: #ff5722; font-weight: bold;">
                            ${row['discounted_price']}
                            <span style="font-size: 16px; color: #999; text-decoration: line-through; margin-left: 5px;">
                                ${row['actual_price']}
                            </span>
                        </p>
                        <div style="display: flex; align-items: center;">
                            <div style="color: #ffd700; font-size: 17px;">  <!-- Color for gold stars -->
                                {row['rating']}
                                {"★" * int(row['rating']) + "☆" * (5 - int(row['rating']))}
                                <span style="font-size: 15px; color: #999;">
                                {"(" + str(int(row['rating_count'])) + " ratings)"}
                            </div>
                        </div>
                    </div>
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )


elif user_query != "" and len(top_products) == 0:
    st.write(f"No matching products found for \"{user_query}\".")
