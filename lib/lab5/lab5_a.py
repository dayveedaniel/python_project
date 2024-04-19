from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, mean

spark = SparkSession.builder.appName("lb5").getOrCreate()

df = spark.read.csv("brooklyn_sales_map.csv", header=True, inferSchema=True)
df = df.filter(df["sale_price"] != 0)

task1 = df.withColumn("price_deviation", df["sale_price"] -
                      df.select(mean("sale_price")).collect()[0][0])
task1 = task1.select("sale_price", "price_deviation")
task1.sample(withReplacement=False, fraction=20 / df.count()).show(10)

task2 = df.groupBy("year_built").agg(avg("gross_sqft").alias("avg_gross_sqft"))
task2.show(10)

task3 = df.groupBy("neighborhood",
                   "building_class_category").agg(avg("sale_price").alias("avg_sale_price"))
task3.show(10)

task4 = df.filter((df["year_built"] <= 2000) & (df["year_built"] != 0))
task4 = task4.select("year_built")
task4.show(10)

spark.stop()

