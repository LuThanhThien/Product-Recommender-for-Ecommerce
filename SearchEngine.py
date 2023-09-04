import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from spellchecker import SpellChecker
import nltk
from nltk.tokenize import word_tokenize
import emoji, contractions
import string
import re


class TFIDFSearch:
    def __init__(self):
        self.input_path = None
        self.df = pd.DataFrame()
        self.description_df = pd.DataFrame()
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.top_ids = None
        self.top_similarities = None
        self.top_number = None
        self.query = None
        self.word_bank = None

    # PREPROCESSING
    @staticmethod
    def merge_dataframe(list_data, on=None, suffixes=('_x', '_y')):
        df_merge = list_data[0]
        for df in list_data[1:]:
            df_merge = df_merge.merge(df, on=on, how='left', suffixes=suffixes)
        return df_merge

    @staticmethod
    def text_cleaning(text):
        # Lowercase the text
        text = text.lower()

        # Expand contractions
        text = contractions.fix(text)

        # Remove numbers, special characters, and patterns
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation

        # Split words containing non-letter characters and keep only letter parts
        words = text.split()
        cleaned_words = []
        for word in words:
            parts = re.findall(r'\b[a-zA-Z]+\b', word)
            cleaned_words.extend(parts)

        # Combine cleaned words into a single string
        text = " ".join(cleaned_words)

        # Remove emoji and emoticons
        text = ''.join(c for c in text if c not in emoji.EMOJI_DATA)
        emoticon_pattern = (r'(:-?\))|(:\))|(:-\()|(:\()|(:-?D)|(:D)|(:-?])|(:])|(:-?\[)|(:\[)|(:-?p)|(:p)|(:-?['
                            r'|/\\])|(:[|/\\])')
        text = re.sub(emoticon_pattern, '', text)

        # Remove extra spaces
        text = " ".join(text.split())

        return text

    def descriptions(self, save=False, path=''):
        description_list = self.df['product_name'].str.cat([self.df['about_product'],
                                                            self.df['category_1'], self.df['category_2']], sep=' ')

        description_list = list(description_list)

        for i in range(len(description_list)):
            description_list[i] = self.text_cleaning(description_list[i])

        self.description_df = pd.DataFrame(description_list)
        self.description_df = self.description_df.rename(columns={0: 'description'})
        self.description_df['product_id'] = self.df['product_id']
        new_df = self.merge_dataframe([self.df, self.description_df], on='product_id')
        if save:
            new_df.to_csv(f'{path}\\amazon-product-web.csv', index=False)

    def create_word_bank(self, save=False, path=''):
        # initialize NLTK's tokenizer
        # nltk.download('punkt')

        # create a set to store the unique words
        self.word_bank = set()

        # tokenize the text data and add unique words to the word bank
        for text in self.description_df['description'].to_list():
            words = word_tokenize(text)
            self.word_bank.update(words)

        # save the word bank to a text file with UTF-8 encoding
        if save:
            with open(f'{path}\\word_bank.txt', 'w', encoding='utf-8') as file:
                for word in self.word_bank:
                    file.write(word + '\n')

    def preprocessing(self, df, path):
        self.df = df
        self.descriptions(save=True, path=path)
        print('Saved descriptions')
        self.create_word_bank(save=True, path=path)
        print('Saved word bank')
        print('Finished.')

    # fit_transform data
    def fit_transform(self, dataFrame, input_path):
        self.df = dataFrame
        self.input_path = input_path
        self.description_df['description'] = self.df['description']
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.description_df['description'])

    def call_word_bank(self):
        with open(f'{self.input_path}\\word_bank.txt', 'r', encoding='utf-8') as file:
            self.word_bank = file.read()

    def correct_spellings(self):
        spell = SpellChecker()

        corrected_text = []
        words = self.query.split()

        if self.word_bank is None:
            self.call_word_bank()

        for word in words:
            if word not in self.word_bank:
                corrected_word = spell.correction(word)
            else:
                corrected_word = word

            corrected_text.append(corrected_word)

        if None in corrected_text:
            return str()

        return ' '.join(corrected_text)

    # fine top similarity
    def search_query(self, user_query, top_number=None):
        self.top_number = top_number
        self.query = user_query
        self.query = self.correct_spellings()

        # Preprocess and vectorize the search query
        query_vector = self.tfidf_vectorizer.transform([self.query])

        # Calculate cosine similarity between the query and all products
        cosine_similarities = linear_kernel(query_vector, self.tfidf_matrix).flatten()
        cosine_indices = cosine_similarities.argsort()[::-1]

        if top_number is None:
            top_number = len(cosine_indices)

        self.top_ids = cosine_indices[:top_number]
        self.top_similarities = [cosine_similarities[i] for i in self.top_ids]

        # Retrieve and display the top N similar products
        self.top_ids = [self.df['product_id'][i] for i in self.top_ids]

        return self.top_ids, self.top_similarities

    # return dataframe with rank

    def search_result(self, threshold=0):
        # Filter products based on the threshold
        self.top_ids = [self.top_ids[i] for i in range(self.top_number) if self.top_similarities[i] > threshold]

        df_top_products = self.df.loc[self.df.product_id.isin(self.top_ids)]

        # Create a dictionary that maps each product ID to its position in the top_products list
        order_dict = {product_id: index for index, product_id in enumerate(self.top_ids)}

        # Create a new column in the df_top_products DataFrame that contains the position of each product ID
        df_top_products['order'] = df_top_products.product_id.map(order_dict)

        # Sort the df_top_products DataFrame by the 'order' column
        df_top_products = df_top_products.iloc[np.argsort(df_top_products.order)]

        # Drop the 'order' column from the resulting DataFrame
        df_top_products = df_top_products.drop('order', axis=1)

        return df_top_products


class TopProduct:
    def __init__(self, dataFrame):
        self.df = dataFrame


def main():
    input_path = 'inputs\\data\\amazon'
    output_path = '..\\outputs\\data\\amazon'
    df = pd.read_csv(f'{output_path}\\amazon-product.csv')
    engine = TFIDFSearch()
    engine.preprocessing(df, path=output_path)


# if __name__ == '__main__':
#     main()
