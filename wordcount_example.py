from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("WordCountExample") \
    .getOrCreate()

# Sample data
data = [
    "Hello Spark",
    "Hello World",
    "Spark is great"
]

# Create RDD from data
rdd = spark.sparkContext.parallelize(data)

# Perform WordCount
word_counts = rdd.flatMap(lambda line: line.split(" ")) \
                 .map(lambda word: (word, 1)) \
                 .reduceByKey(lambda a, b: a + b)

# Collect and print results
print("Word Count Results:")
for word, count in word_counts.collect():
    print(f"{word}: {count}")

# Stop SparkSession
spark.stop()


