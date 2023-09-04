SELECT * 
FROM [dbo].[Sales_Amazon_Product]

SELECT
	product_id, product_name
INTO amazon_product
FROM [dbo].[Sales_Amazon_Product]
GROUP BY
	product_id, product_name

SELECT CONCAT('https://www.amazon.in/dp/', product_id) as urls,
	product_id
FROM amazon_product