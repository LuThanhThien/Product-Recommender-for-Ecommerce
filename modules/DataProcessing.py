import numpy as np
import pandas as pd
import time
import os


class DataCleaner:
    def __init__(self, data, recency_days=15, weight_decay=0.5,
                 view_weight=0.1, cart_weight=0.4, remove_from_cart_weight=-0.3, purchase_weight=1.0):
        self.data = data
        self.product = pd.DataFrame(columns=['product_id'])
        self.user = pd.DataFrame(columns=['user_id'])
        self.interaction = pd.DataFrame(columns=['product_id', 'user_id'])
        self.feature = pd.DataFrame(columns=['product_id', 'user_id', 'interaction_score'])
        self.event_name = ['view', 'cart', 'remove_from_cart', 'purchase']
        self.recency_days = recency_days
        self.weight_decay = weight_decay
        self.view_weight = view_weight
        self.cart_weight = cart_weight
        self.remove_from_cart_weight = remove_from_cart_weight
        self.purchase_weight = purchase_weight

    @staticmethod
    def print_end():
        print('-------------------------------------------------------------')

    @staticmethod
    def merge_dataframe(list_data, on=None, suffixes=('_x', '_y')):
        df_merge = list_data[0]
        for df in list_data[1:]:
            df_merge = df_merge.merge(df, on=on, how='left', suffixes=suffixes)
        return df_merge

    # CLEAN RAW DATA
    # ---------------------------------------------------------------------
    # drop negative rows
    def drop_duplicates(self):
        print('Drop duplicates processing...')
        start = time.time()
        num_duplicates = len(self.data[self.data.duplicated(keep=False)])
        if num_duplicates > 0:
            print('Found {:d} duplicate rows'.format(num_duplicates))
            # Drop duplicates
            self.data = self.data.drop_duplicates()
        else:
            print('Found no duplicate.')
        end = time.time()
        print('End drop duplicates. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    # drop negative prices
    def drop_neg_price(self):
        print('Drop negative price processing...')
        start = time.time()
        num_neg = len(self.data.loc[self.data.price < 0])
        if num_neg > 0:
            print('Found {:d} products with negative price'.format(num_neg))
            # Drop neg price
            self.data = self.data.loc[self.data['price'] >= 0]
        else:
            print('Found no negative price.')
        end = time.time()
        print('End drop negative price. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    # split category_code
    def split_category_code(self):
        print('Split category code processing...')
        start = time.time()
        num_taxonomies = self.data['category_code'].nunique()
        print('Found {:d} category codes'.format(num_taxonomies))

        # Split by the first period (n=1)
        self.data[['main_category', 'sub_category']] = self.data['category_code'].str.split('.', n=1, expand=True)

        # Rearrange the columns
        print('Rearranged columns processing...')
        self.data = self.data.drop(columns=['category_code'])
        new_order = ['event_time', 'event_type', 'product_id', 'category_id', 'main_category', 'sub_category', 'brand',
                     'price', 'user_id', 'user_session']
        self.data = self.data[new_order]

        end = time.time()

        # There are many products have no code, the percentage are:
        no_categoryCode = self.data.drop_duplicates(subset='product_id')['main_category'].isna().sum()
        category_counts = self.data.drop_duplicates(subset='product_id'). \
            groupby(['main_category', 'sub_category']).size().reset_index(name='count')
        have_categoryCode = category_counts['count'].sum()

        if 100 * have_categoryCode / (have_categoryCode + no_categoryCode) < 50:
            print(f'''WARNING: Found column with missing values: 
        Column name: category_code
        Number of NULL rows: {no_categoryCode} missing values/{no_categoryCode + have_categoryCode} total
            ''')

        print('Percentage of products have category_code: {0:.2f}%'.format(
            100 * have_categoryCode / (have_categoryCode + no_categoryCode)))
        print('Percentage of products have NO category_code: {0:.2f}%'.format(
            100 * no_categoryCode / (have_categoryCode + no_categoryCode)))

        print('End split category code. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    # fill products with missing brands
    def fill_brand(self):
        print('Fill missing brand processing...')
        start = time.time()
        num_brands = self.data.groupby('product_id')['brand'].nunique(). \
            sort_values(ascending=False).reset_index().iloc[0]['brand']
        # print('Maximum number of brand a product has: ',num_brands)

        # in case the product has various brands throughout the period: fill the NaN with brand from this table
        if int(num_brands) > 1:

            product_brands = self.data.groupby('product_id')['brand'].unique().reset_index()
            print(f'Found {len(product_brands[product_brands.brand.apply(lambda x: len(x) > 1)])}\
                    products with multiple brands...')

            # if brand has 2 brands, choose the one that is not null
            def clean_brand(brand_list):
                for brand in brand_list:
                    if brand and brand != 'nan':
                        return brand
                return None

            product_brands['brand'] = product_brands['brand'].apply(clean_brand)

            def fill_brand(df, product_brands_df):
                df = df.merge(product_brands_df[['product_id', 'brand']], on='product_id', how='left',
                              suffixes=('', '_new'))
                df['brand'] = df['brand_new'].fillna(df['brand'])
                df.drop(columns=['brand_new'], inplace=True)
                print('Filled successfully')

            fill_brand(self.data, product_brands)

        else:
            print(f'Found no products with multiple brands.')
        end = time.time()

        num_brands = self.data['brand'].nunique()
        print('Total number of brands: ', num_brands)

        print('End fill missing brand. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    # drop cart duplicates
    def drop_cart_duplicates(self):
        print('Drop duplicates in cart...')
        start = time.time()

        # Drop cart duplicates
        before_data = len(self.data)
        cart_data = self.data[self.data.event_type == 'cart']

        # show duplicates
        # print(cart_data[cart_data.duplicated(subset=['product_id', 'user_id', 'user_session'], keep=False)].\
        #       sort_values('user_id').head(10))

        cart_duplicates = len(cart_data[cart_data. \
                              duplicated(subset=['product_id', 'user_id', 'user_session'], keep=False)])
        print('Found {} duplicates'.format(cart_duplicates))

        if cart_duplicates > 0:
            self.data = self.data[self.data.event_type != 'cart']
            cart_data.drop_duplicates(subset=['product_id', 'user_id', 'user_session'], keep='first', inplace=True)

            # check if exist duplicates
            # print(cart_data[cart_data.duplicated(subset=['product_id', 'user_id', 'user_session'], keep=False)].\
            #       sort_values('user_id').head(10))

            # union
            self.data = pd.concat([cart_data, self.data])
            print('Data from {} rows reduced to {} rows'.format(before_data, len(self.data)))

        else:
            print('Found no cart duplicate.')

        end = time.time()

        print('End drop cart duplicates. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    def drop_remove_cart_duplicates(self):
        print('Drop duplicates in remove_from_cart...')
        start = time.time()

        # Drop cart duplicates
        before_data = len(self.data)
        cart_data = self.data[self.data.event_type == 'remove_from_cart']

        # show duplicates
        # print(cart_data[cart_data.duplicated(subset=['product_id', 'user_id', 'user_session'], keep=False)].\
        #       sort_values('user_id').head(10))

        cart_duplicates = len(cart_data[cart_data. \
                              duplicated(subset=['product_id', 'user_id', 'user_session'], keep=False)])
        print('Found {} duplicates'.format(cart_duplicates))

        if cart_duplicates > 0:
            self.data = self.data[self.data.event_type != 'cart']
            cart_data.drop_duplicates(subset=['product_id', 'user_id', 'user_session'], keep='first', inplace=True)

            # check if exist duplicates
            # print(cart_data[cart_data.duplicated(subset=['product_id', 'user_id', 'user_session'], keep=False)].\
            #       sort_values('user_id').head(10))

            # union
            self.data = pd.concat([cart_data, self.data])
            print('Data from {} rows reduced to {} rows'.format(before_data, len(self.data)))

        else:
            print('Found no remove_from_cart duplicate.')

        end = time.time()

        print('End drop remove_from_cart duplicates. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    # split datetime
    def split_datetime(self):
        print('Spit datetime processing...')
        start = time.time()

        # Split into date and time
        self.data['event_time'] = pd.to_datetime(self.data['event_time'])
        self.data['date'] = pd.to_datetime(self.data['event_time'].dt.date)
        self.data['time'] = self.data['event_time'].dt.time

        print('Add weekday column processing...')
        self.data['weekday'] = self.data['date'].dt.weekday

        print('Drop event_time column processing...')
        self.data.drop('event_time', axis='columns')

        self.data = self.data[
            ['date', 'weekday', 'time', 'event_type', 'product_id', 'category_id', 'main_category', 'sub_category',
             'brand', 'price', 'user_id', 'user_session']]

        self.data.sort_values(by=['date', 'time'])
        self.data['date'] = pd.to_datetime(self.data['date'])

        end = time.time()

        print('End split datetime. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    # END CLEAN RAW DATA
    # ---------------------------------------------------------------------

    # FEATURE ENGINEERING
    # ---------------------------------------------------------------------
    # relative price
    def relative_price(self):
        print('Calculating relative price...')
        start = time.time()

        # calculate percentiles_IQR
        percentiles_IQR = self.data.groupby('category_id')['price']. \
            quantile([0.25, 0.5, 0.75]).unstack().reset_index()
        percentiles_IQR.rename(columns={0.25: 'Q1', 0.5: 'median', 0.75: 'Q3'}, inplace=True)
        percentiles_IQR['IQR'] = percentiles_IQR['Q3'] - percentiles_IQR['Q1']

        # calculate relative price by price-median/IQR
        self.data = self.data.merge(percentiles_IQR, on='category_id', how='left')
        self.data['relative_price'] = np.where(self.data['IQR'] == 0,
                                               0, (self.data['price'] - self.data['median']) / self.data['IQR'])
        self.data['relative_price'] = self.data['relative_price'].apply(
            lambda x: max(min(x, 10), -10))  # avoid exceeding
        self.data.drop(columns=['Q1', 'Q3', 'median', 'IQR'], axis='columns', inplace=True)

        # rearrange
        self.data = self.data[
            ['date', 'weekday', 'time', 'event_type', 'product_id', 'category_id', 'main_category', 'sub_category',
             'brand', 'price', 'relative_price', 'user_id', 'user_session']]
        end = time.time()
        # print(self.data.isna().sum())
        print('End calculating relative price. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    # create product table
    def create_product_table(self):
        print('CREATING PRODUCT TABLE...')
        start = time.time()

        # feature engineering
        print('Basic features processing...')
        self.product = self.data.groupby('product_id').agg(
            first_date=('date', 'min'),
            last_date=('date', 'max'),
            category_id=('category_id', 'first'),
            avg_price=('price', 'mean'),
            relative_price=('relative_price', 'mean'),
            views=('event_type', lambda x: (x == 'view').sum()),
            carts=('event_type', lambda x: (x == 'cart').sum()),
            remove_from_carts=('event_type', lambda x: (x == 'remove_from_cart').sum()),
            purchases=('event_type', lambda x: (x == 'purchase').sum()),
        ).reset_index()

        print('Interaction rates processing...')
        # calculate rate
        self.product['cart_per_view'] = 100 * np.where(self.product['views'] == 0, self.product['carts'],
                                                       self.product['carts'] / self.product['views'])
        self.product['purchase_per_view'] = 100 * np.where(self.product['views'] == 0, self.product['purchases'],
                                                           self.product['purchases'] / self.product['views'])
        self.product['remove_per_cart'] = 100 * np.where(self.product['carts'] == 0, self.product['remove_from_carts'],
                                                         self.product['remove_from_carts'] / self.product['carts'])
        self.product['purchase_per_cart'] = 100 * np.where(self.product['carts'] == 0, self.product['purchases'],
                                                           self.product['purchases'] / self.product['carts'])

        # by weighted mean
        total_views = self.product[['views']].sum().sum()
        total_carts = self.product[['carts']].sum().sum()
        total_removes = self.product[['remove_from_carts']].sum().sum()
        total_purchases = self.product[['purchases']].sum().sum()

        # rate by weights
        self.product['cart_per_view'] = self.product['cart_per_view'] \
                                        * (self.product['views'] / total_views)
        self.product['purchase_per_view'] = self.product['purchase_per_view'] \
                                            * (self.product['views'] / total_views)
        self.product['remove_per_cart'] = self.product['remove_per_cart'] \
                                          * ((self.product['carts']) / total_carts)
        self.product['purchase_per_cart'] = self.product['purchase_per_cart'] \
                                            * ((self.product['carts']) / total_carts)

        # normalize rates by min-max
        rate_columns = ['cart_per_view', 'purchase_per_view', 'remove_per_cart', 'purchase_per_cart']
        for col in rate_columns:
            min_value = self.product[col].min()
            max_value = self.product[col].max()
            self.product[col] = (self.product[col] - min_value) / (max_value - min_value)
        end = time.time()

        print('Create product table successfully. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    # user interaction processing
    def create_user_interaction(self):
        # count events
        views = self.data.loc[self.data.event_type == 'view'].groupby('user_id').event_type.count().reset_index()
        views.rename(columns={'event_type': 'views'}, inplace=True)
        carts = self.data.loc[self.data.event_type == 'cart'].groupby('user_id').event_type.count().reset_index()
        carts.rename(columns={'event_type': 'carts'}, inplace=True)
        removes = self.data.loc[self.data.event_type == 'remove_from_cart'].groupby(
            'user_id').event_type.count().reset_index()
        removes.rename(columns={'event_type': 'remove_from_carts'}, inplace=True)
        purchases = self.data.loc[self.data.event_type == 'purchase'].groupby(
            'user_id').event_type.count().reset_index()
        purchases.rename(columns={'event_type': 'purchases'}, inplace=True)
        sumEvent_mask = self.merge_dataframe([views, carts, removes, purchases], on='user_id')
        sumEvent_mask.fillna(0, inplace=True)

        # average price
        viewPrice_mask = self.data.loc[self.data.event_type == 'view'].groupby(['user_id']).price.mean().reset_index()
        viewPrice_mask.rename(columns={'price': 'avg_view_price'}, inplace=True)
        purchasePrice_mask = self.data.loc[self.data.event_type == 'cart'].groupby(
            ['user_id']).price.mean().reset_index()
        purchasePrice_mask.rename(columns={'price': 'avg_purchase_price'}, inplace=True)
        avgPrice_mask = self.merge_dataframe([viewPrice_mask, purchasePrice_mask], on='user_id')
        avgPrice_mask.fillna(0, inplace=True)

        # average price
        viewRelative_mask = self.data.loc[self.data.event_type == 'view'].groupby(
            ['user_id']).relative_price.mean().reset_index()
        viewRelative_mask.rename(columns={'relative_price': 'avg_view_relative_price'}, inplace=True)
        purchaseRelative_mask = self.data.loc[self.data.event_type == 'cart'].groupby(
            ['user_id']).relative_price.mean().reset_index()
        purchaseRelative_mask.rename(columns={'relative_price': 'avg_purchase_relative_price'}, inplace=True)
        avgRelative_mask = self.merge_dataframe([viewRelative_mask, purchaseRelative_mask], on='user_id')
        avgRelative_mask.fillna(0, inplace=True)

        # distinct product counts by event
        viewCount_mask = self.data.loc[self.data.event_type == 'view'].groupby(
            ['user_id']).product_id.nunique().reset_index()
        viewCount_mask.rename(columns={'product_id': 'distinct_view_product'}, inplace=True)
        cartCount_mask = self.data.loc[self.data.event_type == 'cart'].groupby(
            ['user_id']).product_id.nunique().reset_index()
        cartCount_mask.rename(columns={'product_id': 'distinct_cart_product'}, inplace=True)
        removeCount_mask = self.data.loc[self.data.event_type == 'remove_from_cart'].groupby(
            ['user_id']).product_id.nunique().reset_index()
        removeCount_mask.rename(columns={'product_id': 'distinct_remove_product'}, inplace=True)
        purchaseCount_mask = self.data.loc[self.data.event_type == 'purchase'].groupby(
            ['user_id']).product_id.nunique().reset_index()
        purchaseCount_mask.rename(columns={'product_id': 'distinct_purchase_product'}, inplace=True)
        eventCount_mask = self.merge_dataframe([viewCount_mask, cartCount_mask, removeCount_mask, purchaseCount_mask],
                                               on='user_id')
        eventCount_mask.fillna(0, inplace=True)

        return self.merge_dataframe([sumEvent_mask, avgPrice_mask, avgRelative_mask, eventCount_mask, ], on='user_id')

    # create user table
    def create_user_table(self):
        print('CREATING USER TABLE...')
        start = time.time()

        # feature engineering
        print('Basic features processing...')
        self.user = self.data.groupby('user_id').agg(
            first_date=('date', 'min'),
            last_date=('date', 'max'),
        ).reset_index()

        self.user['first_date'] = pd.to_datetime(self.user['first_date'])
        self.user['last_date'] = pd.to_datetime(self.user['last_date'])

        print('Interaction features processing...')
        # create event features
        self.user = self.merge_dataframe([self.user, self.create_user_interaction()], on='user_id')

        print('Interaction rates processing...')
        # calculate rate
        self.user['cart_per_view'] = 100 * np.where(self.user['views'] == 0, self.user['carts'],
                                                    self.user['carts'] / self.user['views'])
        self.user['purchase_per_view'] = 100 * np.where(self.user['views'] == 0, self.user['purchases'],
                                                        self.user['purchases'] / self.user['views'])
        self.user['remove_per_cart'] = 100 * np.where(self.user['carts'] == 0, self.user['remove_from_carts'],
                                                      self.user['remove_from_carts'] / self.user['carts'])
        self.user['purchase_per_cart'] = 100 * np.where(self.user['carts'] == 0, self.user['purchases'],
                                                        self.user['purchases'] / self.user['carts'])
        self.user.fillna(0, inplace=True)

        # by weighted mean
        total_views = self.user[['views']].sum().sum()
        total_carts = self.user[['carts']].sum().sum()
        total_removes = self.user[['remove_from_carts']].sum().sum()
        total_purchases = self.user[['purchases']].sum().sum()

        # rate by weights
        self.user['cart_per_view'] = self.user['cart_per_view'] \
                                     * (self.user['views'] / total_views)
        self.user['purchase_per_view'] = self.user['purchase_per_view'] \
                                         * (self.user['views'] / total_views)
        self.user['remove_per_cart'] = self.user['remove_per_cart'] \
                                       * ((self.user['carts']) / total_carts)
        self.user['purchase_per_cart'] = self.user['purchase_per_cart'] \
                                         * ((self.user['carts']) / total_carts)

        # normalize rates by min-max
        rate_columns = ['cart_per_view', 'purchase_per_view', 'remove_per_cart', 'purchase_per_cart']
        for col in rate_columns:
            min_value = self.user[col].min()
            max_value = self.user[col].max()
            self.user[col] = (self.user[col] - min_value) / (max_value - min_value)

        # change type
        print('Check datatypes processing...')
        columns = self.user.columns
        user_type = ['int64', 'date', 'date',
                     'int64', 'int64', 'int64', 'int64',
                     'float', 'float', 'float', 'float',
                     'int64', 'int64', 'int64', 'int64',
                     'float', 'float', 'float', 'float']
        for i, col in enumerate(columns):
            if i in [1, 2]:
                pass  # change date already
            else:
                self.user[col] = self.user[col].astype(user_type[i])

        end = time.time()
        print('Create user table successfully. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    # for interaction dataset
    def calculate_recency(self):
        print('Calculating recency processing...')

        start = time.time()
        # Get the most recent event in the data_recency dataset
        data_recency = self.data.copy()
        data_recency['date'] = pd.to_datetime(data_recency['date'])
        last_date = data_recency['date'].max()

        # Calculate the recency of each event in terms of days
        data_recency['recency'] = (last_date - data_recency['date']).dt.total_seconds() / (24 * 60 * 60)

        # Half-life decay function, the value of an event with weight_decay after recency_days days
        data_recency['recency_coef'] = np.exp(np.log(self.weight_decay) * data_recency['recency'] / self.recency_days)

        # Drop the 'date' column if needed for real-time system
        data_recency.drop(columns=['date', 'time'], inplace=True)

        end = time.time()
        print('Calculating recency successfully. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

        return data_recency

    # create interaction table
    def create_interaction_table(self):
        print('CREATE USER-PRODUCT INTERACTION TABLE...')
        start = time.time()

        data_recency = self.calculate_recency()
        print('Calculating basic interactions...')
        # initial dataframe
        self.interaction = pd.DataFrame(columns=['user_id', 'product_id'])
        self.interaction.user_id = data_recency.user_id
        self.interaction.product_id = data_recency.product_id
        self.interaction.drop_duplicates(inplace=True)

        # for every event, calculate the interaction score for each user-product pair
        event_names = ['view', 'cart', 'remove_from_cart', 'purchase']
        event_weights = [self.view_weight, self.cart_weight, self.remove_from_cart_weight, self.purchase_weight]
        for name in event_names:
            df = data_recency.loc[data_recency['event_type'] == name].groupby(
                ['user_id', 'product_id']).recency_coef.sum().reset_index()
            df.rename(columns={'recency_coef': f'{name}s'}, inplace=True)
            self.interaction = self.merge_dataframe([self.interaction, df], on=['user_id', 'product_id'])

        # fill NaN with 0 (0 interaction score)
        self.interaction.fillna(value=0, inplace=True)

        print('Calculating interaction scores...')
        # calculate the interaction overall score for user-product pair by weight
        self.interaction['interaction_score'] = 0
        for weight, name in zip(event_weights, event_names):
            self.interaction['interaction_score'] += weight * self.interaction[f'{name}s']

        # max 100, min 0
        self.interaction['interaction_score'] = self.interaction['interaction_score'].apply(lambda x: max(0, x))
        self.interaction['interaction_score'] = self.interaction['interaction_score'].apply(lambda x: min(100, x))

        self.interaction.sort_values(by='interaction_score', ascending=False, inplace=True)

        # drop columns
        for name in event_names:
            self.interaction.drop(columns=f'{name}s', inplace=True)
        end = time.time()

        print('Create interaction table successfully. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    def create_train_table(self):
        print('CREATE TRAINING TABLE...')
        start = time.time()
        self.feature = self.merge_dataframe([self.interaction, self.user], on='user_id', suffixes=('', '_user'))
        self.feature = self.merge_dataframe([self.feature, self.product], on='product_id',
                                            suffixes=('_user', '_product'))
        self.feature.drop(columns=['first_date_user', 'last_date_user', 'first_date_product', 'last_date_product'],
                          inplace=True)
        end = time.time()

        print('Create training table successfully. Finished in {0:.3f}s.'.format(end - start))
        self.print_end()

    def CleanData(self, save=False, name='data_clean', save_path='clean-data'):
        print('CLEANING DATA PROCESSING...')
        time.sleep(3)
        start = time.time()
        self.drop_duplicates()
        self.drop_neg_price()
        self.split_category_code()
        self.fill_brand()
        self.drop_cart_duplicates()
        self.drop_remove_cart_duplicates()
        self.split_datetime()
        end = time.time()

        if save:
            print(f"Saving to '{save_path}\\{name}.csv'...")
            self.data.to_csv(f'{save_path}\\{name}.csv')

        print('Clean data successfully.\nFinished in {0:.3f}s.'.format(end - start))
        self.print_end()
        return self.data

    def FeatureEngineering(self, save=False, name='data_train', save_path='clean-data'):
        if 'date' not in self.data.columns:
            raise AttributeError('''Raw data has not cleaned! You have to call CleanData() for cleaning first.''')

        print('FEATURE ENGINEERING PROCESSING...')
        time.sleep(3)
        start = time.time()

        self.relative_price()
        self.create_product_table()
        self.create_user_table()
        self.create_interaction_table()
        self.create_train_table()
        end = time.time()

        if save:
            print(f"Saving to '{save_path}\\{name}.csv'...")
            self.feature.to_csv(f'{save_path}\\{name}.csv')
        print('Extract features successfully.\nFinished in {0:.3f}s.'.format(end - start))
        self.print_end()
        return self.product, self.user, self.interaction, self.feature


def main():
    input_path = '..\\data\\comestic\\raw'
    output_path = '..\\data\\comestic\\clean'
    input_filenames = []
    output_filenames = []

    for _, _, filename in os.walk(input_path):
        input_filenames.append(filename)
    for _, _, filename in os.walk(output_path):
        output_filenames.append(filename)

    input_filenames = [x for x in [input_filenames[0][i].split('.')[0] for i in range(len(input_filenames[0]))]]
    output_filenames = output_filenames[0]

    # print(filenames)
    for file in input_filenames:
        if f'{file}-clean.csv' in output_filenames:
            print(f"'{file}-clean.csv' exists\n")
            data_clean = pd.read_csv(f"{output_path}\\{file}-clean.csv")
        else:
            print(f"Processing '{file}-clean.csv'")
            data = pd.read_csv(f"{input_path}\\{file}.csv")
            cleaner = DataCleaner(data)
            data_clean = cleaner.CleanData(save=True, name=f'{file}-clean', save_path=output_path)

        if f'{file}-feature.csv' in output_filenames:
            print(f"'{file}-feature.csv' exists\n")
        else:
            print(f"Processing '{file}-clean.csv'")
            cleaner = DataCleaner(data_clean)
            data_feature = cleaner.FeatureEngineering(save=True, name=f'{file}-feature', save_path=output_path)


if __name__ == '__main__':
    main()