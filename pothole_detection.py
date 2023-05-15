import os
import boto3
from io import BytesIO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.types import *
from pyspark.sql.functions import col, when
import matplotlib
import matplotlib.pyplot as plt
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import random
#Creating a user to access S3 and save the output
os.environ['AWS_ACCESS_KEY_ID'] = 'AKIA55LOFPGMA5LWZZHA'
os.environ['AWS_SECRET_ACCESS_KEY'] = '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'
# Initialize Spark session
spark = SparkSession.builder \
    .appName("Pothole Detection") \
    .getOrCreate()

# Define a function to load images from S3 bucket and create a PySpark DataFrame
def load_images_from_s3(bucket_name, folder_prefix, pothole=True):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    images = []
    image_paths = []
    for obj in bucket.objects.filter(Prefix=folder_prefix):
        if obj.key.endswith('.jpg'):
            img_file = BytesIO()
            bucket.download_fileobj(obj.key, img_file)
            img = Image.open(img_file).convert('L').resize((64, 64))
            img_data = np.array(img).flatten().tolist()
            label = 1 if pothole else 0
            images.append((img_data, label))
            image_paths.append(obj.key)

    schema = StructType([
        StructField("features", ArrayType(IntegerType()), True),
        StructField("label", IntegerType(), True)
    ])

    return spark.createDataFrame(images, schema), image_paths

# Load the dataset from S3 (assuming a folder structure with images in 'potholes' and 'no_potholes' subfolders)
bucket_name = "projectadbms"
potholes_data, potholes_image_paths = load_images_from_s3(bucket_name, "potholes/")
no_potholes_data, no_potholes_image_paths = load_images_from_s3(bucket_name, "no_potholes/", pothole=False)
data = potholes_data.union(no_potholes_data)
image_paths = potholes_image_paths + no_potholes_image_paths


# Split the data into training and test sets
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)

# Convert feature arrays to dense vectors
to_dense = udf(lambda arr: Vectors.dense(arr), VectorUDT())
train_data = train_data.withColumn("features", to_dense(col("features")))
test_data = test_data.withColumn("features", to_dense(col("features")))

# Train a logistic regression model
lr = LogisticRegression(maxIter=10, regParam=0.001, elasticNetParam=0.8)
model = lr.fit(train_data)

# Make predictions on the test set
predictions = model.transform(test_data)

# Evaluate the model's accuracy
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print("Test set accuracy: {:.2f}%".format(accuracy * 100))

from datetime import datetime


def predict_random_image(model, bucket_name, save_folder):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    images = [obj.key for obj in bucket.objects.all() if obj.key.endswith('.jpg')]
    random_image_path = random.choice(images)
    img_file = BytesIO()
    bucket.download_fileobj(random_image_path, img_file)
    
    # Save the original color image for later use
    original_img = Image.open(img_file)
    
    # Resize and convert to grayscale for prediction
    img = original_img.resize((64, 64)).convert('L')
    img_data = np.array(img).flatten().tolist()

    # Create a DataFrame with the image and make prediction
    schema = StructType([
        StructField("features", ArrayType(IntegerType()), True)
    ])
    df = spark.createDataFrame([(img_data,)], schema)
    df = df.withColumn("features", to_dense(col("features")))
    prediction = model.transform(df).first().prediction

    # Generate a timestamp and include it in the filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Define the filename based on the prediction
    filename = "pothole" if prediction == 1 else "no_pothole"
    filename = f'{filename}_{timestamp}.jpg'

    # Save the original color image to a BytesIO object
    img_byte_arr = BytesIO()
    original_img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Save the original color image back to S3
    s3.Bucket(bucket_name).put_object(Key=f'{save_folder}/{filename}', Body=img_byte_arr)





# Predict and save a random image
predict_random_image(model, bucket_name, 'predictions')


# Stop the Spark session
spark.stop()
