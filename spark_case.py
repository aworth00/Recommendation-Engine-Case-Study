import pyspark
from pyspark.sql import SparkSession
import numpy as np
import pandas as pd
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
import pyspark.sql.functions as F
# import seaborn as sns
# import matplotlib.pyplot as plt
import sys
import numpy as np
from surprise import AlgoBase, Dataset, evaluate
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
# import alert_program as ap

spark = SparkSession.builder.getOrCreate()


ratings_df = pd.read_table("data/ratings.dat", delimiter = '::',
                                     names=["user", "movie", "rating",
                                            "timestamp"], engine = 'python')
spark_df = spark.createDataFrame(ratings_df)

spark_df = spark_df.drop("timestamp")
train, test = spark_df.randomSplit([0.8, 0.2], seed=427471138)

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

pandas_df = predictions.toPandas()
pandas_df_clean=pandas_df.fillna(pandas_df.mean())
pandas_df_clean['RMSE']=np.power(pandas_df_clean['rating']-pandas_df_clean['prediction'],2)
RMSE = np.sqrt(sum(pandas_df_clean['RMSE']) / len(pandas_df_clean))

print (RMSE)
