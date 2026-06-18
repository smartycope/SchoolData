USE BikeStores;

SELECT *
FROM dbo.sales.customers;
-- WHERE state = 'CA';


SELECT *
FROM sys.tables;


SELECT s.name AS schema_name, t.name AS table_name
FROM sys.tables t
JOIN sys.schemas s ON t.schema_id = s.schema_id;
-- WHERE t.name = 'brands';
