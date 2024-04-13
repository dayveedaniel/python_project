import warnings

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.mllib.evaluation import MulticlassMetrics

spark = SparkSession.builder.appName("lb5").getOrCreate()
spark.sparkContext.setLogLevel(logLevel="OFF")

log4j = spark._jvm.org.apache.log4j
log4j.LogManager.getRootLogger().setLevel(log4j.Level.OFF)

warnings.filterwarnings('ignore')

data = spark.read.option("delimiter", ";").csv("/kaggle/input/bank-full1",
                                               header=True, inferSchema=True)

# data = data.repartition(1000)
data.show()

numerical_cols = [col for col, dtype in data.dtypes if dtype in ['int', 'double']]
categorical_cols = [col for col, dtype in data.dtypes if dtype == 'string']

indexer = StringIndexer(inputCols=categorical_cols, outputCols=[f"{c}_index" for
                                                                c in categorical_cols], handleInvalid="keep")
encoder = OneHotEncoder(inputCols=[f"{c}_index" for c in categorical_cols],
                        outputCols=[f"{c}_ohe" for c in categorical_cols])
assembler = VectorAssembler(inputCols=numerical_cols, outputCol="numerical")
scaler = StandardScaler(inputCol="numerical", outputCol="numerical_scaled")
final = VectorAssembler(inputCols=[f"{c}_ohe" for c in categorical_cols if c !=
                                   "y"] + ["numerical_scaled"], outputCol="X")
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, final])

train_data, test_data = data.randomSplit([0.8, 0.2], seed=0)
pipeline_model = pipeline.fit(train_data)
train_data = pipeline_model.transform(train_data)
test_data = pipeline_model.transform(test_data)


def model_processing(model, train_data, test_data, paramGrid, model_name,
                     model_params):
    evaluator = MulticlassClassificationEvaluator(labelCol="y_index")
    crossval = CrossValidator(estimator=model,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=3)
    cvModel = crossval.fit(train_data)
    best_model = cvModel.bestModel
    params = best_model.extractParamMap()
    print(model_name)
    print("Best model parameters:")
    for param, value in params.items():
        if param.name in model_params:
            print(f"{param.name}: {value}")
    for i, data in enumerate([train_data, test_data]):
        if i == 0:
            print("Train")
        elif i == 1:
            print("Test")
        predictions = cvModel.transform(data)
        predictionAndTarget = predictions.select("y_index", "prediction")
        metrics_multi = MulticlassMetrics(predictionAndTarget.rdd.map(tuple))
        matrix = metrics_multi.confusionMatrix().toArray()
        print(matrix)
        acc = metrics_multi.accuracy
        print("Accuracy", acc)
        recall = metrics_multi.recall(1.0)
        print("Recall", recall)
        precision = metrics_multi.precision(1.0)
        print("Precision", precision)


lr = LogisticRegression(featuresCol="X", labelCol="y_index")
paramGrid = ParamGridBuilder() \
    .addGrid(lr.maxIter, [10, 100, 1000]) \
    .addGrid(lr.regParam, [0.1, 0.5, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()
model_processing(lr, train_data, test_data, paramGrid, "Logistic Regresson",
                 ["elasticNetParam", "maxIter", "regParam"])
dt = DecisionTreeClassifier(featuresCol="X", labelCol="y_index")
paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [3, 5, 9, 12, 15]) \
    .build()
model_processing(dt, train_data, test_data, paramGrid, "Decision Tree",
                 ["maxDepth"])
rf = RandomForestClassifier(featuresCol="X", labelCol="y_index")
paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [3, 5, 9, 12]) \
    .addGrid(rf.numTrees, [5, 11, 25]) \
    .build()
model_processing(rf, train_data, test_data, paramGrid, "Random Forest",
                 ["maxDepth", "numTrees"])
