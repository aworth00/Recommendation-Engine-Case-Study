from pyspark.sql import SparkSession
import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics
from pyspark.ml.tuning import TrainValidationSplit
from pyspark.sql.types import *
# import alert_program as ap

spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext

# ratings_df = pd.read_table("data/ratings.dat", delimiter = '::',
                                     # names=["user", "movie", "rating",
                                     #        "timestamp"], engine = 'python')
# spark_df = spark.createDataFrame(ratings_df)
# file_path = 'ratings.dat'
# mat = np.fromfile(file_path, count=-1,sep='')
# rdd = sc.parallelize(mat)

rdd = sc.textFile('file:///home/ec2-user/ratings.dat')

def casting(row):
    user, movie, rating, timestamp = row
    return int(user), int(movie), float(rating)

# clean_rdd = rdd.map(lambda row: print(row))
clean_rdd = rdd.map(lambda row: row.split('::')).map(casting)

schema = StructType( [
   StructField('user', IntegerType(), True),
   StructField('movie', IntegerType(), True),
   StructField('rating', FloatType(), True)]
)

ratings_df = spark.createDataFrame(clean_rdd, schema)

train, test = ratings_df.randomSplit([0.8, 0.2], seed=427471138)

als = ALS(
          userCol="user",
          itemCol="movie",
          ratingCol="rating",
          nonnegative=False,
          regParam=0.1,
          rank=10
         )
model = als.fit(train)

predictions = model.transform(test)

evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')

predictions = predictions.dropna()

rmse = evaluator.evaluate(predictions)

# predAndLabel = predictions.select('rating','prediction')
# metrics = RegressionMetrics(predAndLabel)
# print('RMSE = ', metrics.rootMeanSquaredError)

print('RMSE: ', rmse)

# pandas_df = predictions.toPandas()
# pandas_df_clean=pandas_df.fillna(pandas_df.mean())
# pandas_df_clean['RMSE']=np.power(pandas_df_clean['rating']-pandas_df_clean['prediction'],2)
# RMSE = np.sqrt(sum(pandas_df_clean['RMSE']) / len(pandas_df_clean))
#
# print (RMSE)
