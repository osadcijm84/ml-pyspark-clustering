from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler, HashingTF, IDF
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

# Initialize SparkSession
spark = SparkSession.builder \
    .appName("PySparkClustering") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

def run_clustering(data_path="simulated_openfoodfacts_data.jsonl", k=3):
    print(f"Starting PySpark clustering with data from {data_path} and k={k}...")

    # Load data
    # In a real scenario, this would load the actual large dataset
    try:
        df = spark.read.json(data_path)
        print("Data loaded successfully.")
        df.printSchema()
        df.show(5)
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Using dummy data for demonstration.")
        # Fallback to dummy data if loading fails (e.g., file not found)
        data = [
            {"_id": "p1", "product_name": "Product A", "nutriscore_score": 1, "energy_100g": 100.0, "proteins_100g": 5.0, "fat_100g": 10.0, "carbohydrates_100g": 20.0, "ingredients_text": "sugar, salt, water"},
            {"_id": "p2", "product_name": "Product B", "nutriscore_score": 2, "energy_100g": 200.0, "proteins_100g": 10.0, "fat_100g": 15.0, "carbohydrates_100g": 30.0, "ingredients_text": "flour, eggs, milk"},
            {"_id": "p3", "product_name": "Product C", "nutriscore_score": 3, "energy_100g": 150.0, "proteins_100g": 7.0, "fat_100g": 12.0, "carbohydrates_100g": 25.0, "ingredients_text": "rice, chicken, spices"},
            {"_id": "p4", "product_name": "Product D", "nutriscore_score": 1, "energy_100g": 120.0, "proteins_100g": 6.0, "fat_100g": 11.0, "carbohydrates_100g": 22.0, "ingredients_text": "sugar, water, lemon"},
            {"_id": "p5", "product_name": "Product E", "nutriscore_score": 4, "energy_100g": 300.0, "proteins_100g": 15.0, "fat_100g": 20.0, "carbohydrates_100g": 40.0, "ingredients_text": "chocolate, nuts, butter"}
        ]
        df = spark.createDataFrame(data)
        df.printSchema()
        df.show(5)

    # Data Preprocessing
    # Select relevant features and handle missing values
    feature_cols = ["nutriscore_score", "energy_100g", "proteins_100g", "fat_100g", "carbohydrates_100g"]
    
    # Cast columns to appropriate types and fill nulls with 0 (or mean/median in a real scenario)
    for col_name in feature_cols:
        df = df.withColumn(col_name, col(col_name).cast("double"))
        df = df.withColumn(col_name, when(col(col_name).isNull(), 0.0).otherwise(col(col_name)))

    # For text features (ingredients_text), we'll use HashingTF and IDF
    # Fill nulls with empty string for text column
    df = df.withColumn("ingredients_text", when(col("ingredients_text").isNull(), "").otherwise(col("ingredients_text")))

    # Tokenize text
    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer(inputCol="ingredients_text", outputCol="words")
    words_data = tokenizer.transform(df)

    # HashingTF to convert words to feature vectors
    hashing_tf = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=1000)
    featurized_data = hashing_tf.transform(words_data)

    # IDF to re-weight the raw features
    idf = IDF(inputCol="raw_features", outputCol="idf_features")
    idf_model = idf.fit(featurized_data)
    rescaled_data = idf_model.transform(featurized_data)

    # Combine all numerical and text features into a single vector
    assembler = VectorAssembler(
        inputCols=feature_cols + ["idf_features"],
        outputCol="features_raw")
    assembled_data = assembler.transform(rescaled_data)

    # Scale features
    scaler = StandardScaler(inputCol="features_raw", outputCol="features",
                            withStd=True, withMean=False)
    scaler_model = scaler.fit(assembled_data)
    scaled_data = scaler_model.transform(assembled_data)

    # Train K-Means model
    kmeans = KMeans(featuresCol="features", k=k, seed=1)
    model = kmeans.fit(scaled_data)

    # Make predictions
    predictions = model.transform(scaled_data)

    print(f"Clustering completed with {k} clusters.")
    print("Cluster Centers:")
    for center in model.clusterCenters():
        print(center)

    print("Sample predictions:")
    predictions.select("_id", "product_name", "prediction").show(10)

    # Evaluate clustering by computing Within Set Sum of Squared Errors (WSSSE)
    wssse = model.computeCost(scaled_data)
    print(f"Within Set Sum of Squared Errors = {wssse}")

    # Save the model (optional)
    # model.write().overwrite().save("kmeans_model")
    # print("K-Means model saved to kmeans_model")

    # Stop SparkSession
    spark.stop()
    print("SparkSession stopped.")

if __name__ == "__main__":
    # First, simulate data generation
    from simulate_data import simulate_data_loading
    simulated_file = simulate_data_loading(num_records=100)
    
    # Then run clustering on the simulated data
    run_clustering(data_path=simulated_file, k=5)


