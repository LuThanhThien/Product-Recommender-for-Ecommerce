-- * is important query, ** is query for updating original dataset
-- OVERVIEW
SELECT TOP 10 *
FROM [dbo].[Oct-2019];

SELECT COUNT(*)
FROM [dbo].[Oct-2019];

SELECT 
	COUNT(*) as total_events
FROM [dbo].[Oct-2019];

-- check duplicate rows**
DELETE duplicateRows
FROM (
	SELECT
		*, 
		ROW_NUMBER() OVER (
			PARTITION BY 
				date, weekday, time, event_type, product_id, 
				category_id, main_category, sub_category, brand,
				price, user_id, user_session
			ORDER BY (SELECT NULL)) AS rn
	FROM [dbo].[Oct-2019]) AS duplicateRows
WHERE rn > 1

/* 111,600 duplicate rows */

SELECT 
	COUNT(*) as total_events
FROM [dbo].[Oct-2019];

/* 42,418,544 rows*/

-- customer journey
SELECT TOP 10 *
FROM [dbo].[Oct-2019]
WHERE user_id = 555464672;


/* check datetime format */
SELECT *
FROM [dbo].[Oct-2019]
WHERE ISDATE(time) = 0;


/* update the date time format*/
SELECT TOP 10
    product_id,
    category_id,
    main_category,
    sub_category,
    brand,
    CONVERT(VARCHAR(8), time, 108) AS time,
	CONVERT(VARCHAR(10), date, 103) AS date
FROM 
    [dbo].[Oct-2019];

-- PRODUCT TABLES

-- number of product_id
SELECT 
	COUNT(DISTINCT product_id)
FROM [dbo].[Oct-2019];

/* Total products: 166,794 */

-- average price by product_id
SELECT 
	product_id, category_id, 
	main_category, sub_category,
	AVG(price) AS product_avg_price
FROM [dbo].[Oct-2019]
GROUP BY
	product_id,
    category_id,
    main_category,
    sub_category
ORDER BY
	product_avg_price DESC;

-- group by product_id and partition by brand
SELECT 
	product_id, category_id, 
	main_category, sub_category, brand,
	AVG(price) AS product_avg_price
FROM [dbo].[Oct-2019]
GROUP BY
	product_id,
    category_id,
    main_category,
    sub_category,
    brand
ORDER BY 
	product_id DESC;

/* Number of row in GROUP BY only product_id and GROUP BY product_id + brand are different
	Conclusion: there are some product_id have multiple brands*/

-- products with multi-brands
WITH ProductByBrands AS(
	SELECT 
		product_id, category_id, 
		main_category, sub_category, brand,
		AVG(price) AS product_avg_price
	FROM [dbo].[Oct-2019]
	GROUP BY
		product_id,
		category_id,
		main_category,
		sub_category,
		brand)
SELECT 
    COUNT(*) 
FROM (
    SELECT 
        *, 
        COUNT(*) OVER (PARTITION BY product_id) AS number_brands
    FROM ProductByBrands
) AS Subquery
WHERE number_brands = 2 
--ORDER BY
  --  number_brands DESC;

-- table product_id with their brands*
WITH ProductByBrands AS(
	SELECT 
		product_id, category_id, 
		main_category, sub_category, brand,
		AVG(price) AS product_avg_price
	FROM [dbo].[Oct-2019]
	GROUP BY
		product_id,
		category_id,
		main_category,
		sub_category,
		brand)
SELECT DISTINCT
	product_id
	,COALESCE(
		(SELECT TOP 1 brand
		FROM ProductByBrands
		WHERE product_id = t.product_id AND brand IS NOT NULL
		ORDER BY brand DESC), brand
	) AS brand
INTO ProductBrands 
FROM ProductByBrands as t
-- for testing
--WHERE product_id = 21402619 -- this product_id has no brand 
--WHERE product_id = 18301069 -- this product_id brand chages  (2 brands: NaN, hama)
--WHERE product_id = 16600255 -- this product_id brand chages  (3 brands: NaN, jeep, jeepwrangler)
;

-- ProductBrands table
SELECT COUNT(*)
FROM ProductBrands
WHERE brand = ''
--ORDER BY brand
;

/* 123,181 product_id has brand, 43,613 has NULL brand */

-- fill the original table with brands**
UPDATE tabl
SET tabl.brand = ref.brand
FROM [dbo].[Oct-2019] as tabl
JOIN ProductBrands  AS ref ON tabl.product_id = ref.product_id

-- start test of UPDATE
/*
SELECT o.date, o.weekday, o.time, o.event_type,
	o.product_id, o.category_id, o.main_category, 
	o.sub_category, o.brand, o.price, o.user_id, o.user_session
INTO dataNoBrand
FROM [dbo].[Oct-2019] AS o
JOIN ProductBrands  AS n ON o.product_id = n.product_id
WHERE o.brand = '' AND n.brand != ''
ORDER BY
	o.product_id;

SELECT *
FROM dataNoBrand
ORDER BY 
	product_id

UPDATE tabl
SET tabl.brand = ref.brand
FROM dataNoBrand as tabl
JOIN ProductBrands  AS ref ON tabl.product_id = ref.product_id

SELECT *
FROM dataNoBrand
ORDER BY 
	product_id
*/
-- end test of UPDATE

-- Verify the updated values
SELECT TOP 10 * 
FROM [dbo].[Oct-2019];


-- product journey
SELECT 
	*
FROM [dbo].[Oct-2019]
WHERE product_id = 13102543
ORDER BY
	date, time;

-- number of events and abandonment rate
WITH DistinctSession AS (
	SELECT *
	FROM [dbo].[Oct-2019]
	WHERE 
		user_session IN (SELECT DISTINCT user_session FROM [dbo].[Oct-2019])
	)
SELECT 
	product_id, category_id
	,main_category, sub_category
	,AVG(price) AS product_avg_price

	/* total number of views*/
	,SUM(CASE WHEN event_type = 'view' 
			THEN 1 
			ELSE 0 
		END) AS product_total_views
	
	/* total number of carts*/
	,SUM(CASE WHEN event_type = 'cart' 
			THEN 1 
			ELSE 0 
		END) AS product_total_carts
	
	/* total number of purchases*/
	,SUM(CASE WHEN event_type = 'purchase' 
			THEN 1 
			ELSE 0 
		END) AS product_total_purchases
	,CASE
        WHEN SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) > 0
        THEN CAST(SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS FLOAT) / 
             CAST(SUM(CASE WHEN event_type = 'cart' THEN 1 ELSE 0 END) AS FLOAT)
        ELSE 0
    END AS abandonment_rate
FROM DistinctSession
GROUP BY
	product_id
    ,category_id
    ,main_category
    ,sub_category
	,brand
ORDER BY
	product_id, abandonment_rate DESC;





----------------------------------------------------------
SELECT TOP 10 *
FROM (
    SELECT 
        product_id, category_id, 
        main_category, sub_category, brand,
        event_type, price
    FROM [dbo].[Oct-2019]
	WHERE sub_category = 'smartphone'
) AS Otc2019;


SELECT 
	product_id, sub_category, event_type, event_count
FROM (
	SELECT
		product_id, sub_category, event_type,
		COUNT(event_type) AS event_count
		--RANK() OVER (PARTITION BY sub_category ORDER BY COUNT(event_type) DESC) AS max_event_count_rank
	FROM [dbo].[Oct-2019]
	WHERE sub_category = 'smartphone'
	GROUP BY product_id, sub_category, event_type
) AS RankedEvents
GROUP BY event_count
ORDER BY event_count DESC;


WITH MAXEC AS (
	SELECT 
		product_id, sub_category, event_type, event_count
	FROM (
		SELECT
			product_id, sub_category, event_type,
			COUNT(event_type) AS event_count,
			RANK() OVER (PARTITION BY sub_category ORDER BY COUNT(event_type) DESC) AS max_event_count_rank
		FROM [dbo].[Oct-2019]
		WHERE sub_category = 'smartphone'
		GROUP BY product_id, sub_category, event_type
	) AS RankedEvents
	WHERE max_event_count_rank = 1
)
SELECT 
	Oct2019.product_id, Oct2019.category_id, 
	Oct2019.main_category, Oct2019.sub_category, Oct2019.brand,
	-- average price
	AVG(Oct2019.price) AS product_avg_price,
	-- total number of views
	SUM(CASE WHEN Oct2019.event_type = 'view' 
			THEN 1 
			ELSE 0 
		END) AS product_total_views,
	-- total number of carts
	SUM(CASE WHEN Oct2019.event_type = 'cart' 
			THEN 1 
			ELSE 0 
		END) AS product_total_carts,
	-- total number of purchaseS
	SUM(CASE WHEN Oct2019.event_type = 'purchase' 
			THEN 1 
			ELSE 0 
		END) AS product_total_purchases,
	-- calculate popularity score as a weighted average
    (
        0.2 * SUM(CASE WHEN Oct2019.event_type = 'view' THEN 1 ELSE 0 END) +
		0.3 * SUM(CASE WHEN Oct2019.event_type NOT IN ('cart','purchase') THEN 1 ELSE 0 END)  +
        0.5 * SUM(CASE WHEN Oct2019.event_type = 'purchase' THEN 1 ELSE 0 END) 
    )/MAXEC.event_count AS popularity_score
-- Join to apply the max_event_count value
FROM [dbo].[Oct-2019] AS Oct2019
INNER JOIN MAXEC ON Oct2019.product_id = MAXEC.product_id AND Oct2019.sub_category = MAXEC.sub_category
WHERE Oct2019.sub_category = 'smartphone'
GROUP BY
	Oct2019.product_id, Oct2019.category_id, 
	Oct2019.main_category, Oct2019.sub_category, Oct2019.brand
ORDER BY
	product_total_views DESC;

/* Customer Table:
user_id: Unique identifier for each user.
user_first_interaction: The date and time of the user's first interaction.
user_last_interaction: The date and time of the user's last interaction.
user_total_sessions: Total number of sessions for each user.
user_avg_session_duration: Average duration of sessions for each user.
user_avg_price: Average price of products interacted with by the user.
user_most_common_category: The most common category the user interacts with.
user_most_common_brand: The most common brand the user interacts with.
user_total_views: Total number of views made by the user.
user_total_carts: Total number of views made by the user.
user_total_purchase: Total number of purchases made by the user.

Product Table:
product_id: Unique identifier for each product.
category_id: Category of the product.
main_category: Main category of the product.
sub_category: Sub-category of the product.
brand: Brand of the product.
product_avg_price: Average price of the product.
product_total_views: Total number of times the product was viewed.
product_total_carts: Total number of times the product was added to the cart.
product_total_purchases: Total number of times the product was purchased.
first_date: The date and time of the product's first interaction.
last_date: The date and time of the product's last interaction.
product_days_since_last_interaction: Number of days since the product's last interaction.
*/