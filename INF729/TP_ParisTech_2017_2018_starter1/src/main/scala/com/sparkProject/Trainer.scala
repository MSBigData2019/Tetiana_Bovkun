package com.sparkProject


import org.apache.spark.SparkConf
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder, TrainValidationSplit}


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()


    /*******************************************************************************
      *
      *       TP 3
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    val df: DataFrame = spark
      .read
      .option("header", true)  // Use first line of all files as header
      .option("inferSchema", "true") // Try to infer the data types of each column
      .parquet("/Users/tetianabovkun/spark-2.2.0-bin-hadoop2.7/prepared_trainingset")

    //println(s"Total number of rows: ${df.count}")
    //println(s"Number of columns ${df.columns.length}")

    //df.show()
    //df.printSchema()
    //1.a
    val tokenizer = new RegexTokenizer()
       .setPattern("\\W+")
       .setGaps(true)
       .setInputCol("text")
       .setOutputCol("tokens")
    //2.b

    val remover = new StopWordsRemover()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("clean")
    //2.c

    val CV = new CountVectorizer()
      .setInputCol(remover.getOutputCol)
      .setOutputCol("countvect")
      .setMinDF(1)
    //2.d
    val IDF = new IDF()
      .setInputCol(CV.getOutputCol)
      .setOutputCol("tfidf")
    //Question3
    //Q3.e
    val country_indexer = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("countryIndex")
    //Q3.f
    val currency_indexer = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currencyIndex")
    //Q3.g

    val country_encoder = new OneHotEncoder()
      .setInputCol(country_indexer.getOutputCol)
      .setOutputCol("country_onehot")

    val currency_encoder = new OneHotEncoder()
      .setInputCol(currency_indexer.getOutputCol)
      .setOutputCol("currency_onehot")
    //Q4.h
    val assembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal",
        "country_onehot", "currency_onehot"))
      .setOutputCol("features")
    //Q4.i

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThresholds(Array(0.7, 0.3))
      .setTol(1.0e-6)
      .setMaxIter(300)




    //Q4.j
    val pipeline = new Pipeline()
        .setStages(Array(tokenizer, remover, CV, IDF, country_indexer,
          currency_indexer, country_encoder,currency_encoder, assembler, lr))




    //Q4.k
    val Array(df_training, df_test) = df.randomSplit(Array(0.9,0.1))




    //Créer une grille de valeurs à tester pour les hyper-paramètres
    val paramGrid = new ParamGridBuilder()
        .addGrid(CV.minDF, Array(55.0, 75.0, 95.0))
        .addGrid(lr.regParam, Array(1.0e-8, 1.0e-6,1.0e-4,1.0e-2))
        .build()

    //On veut utiliser le f1-score pour comparer les différents modèles en chaque point de la grille
    val evaluator = new MulticlassClassificationEvaluator()
       .setLabelCol("final_status")
       .setPredictionCol("predictions")
       .setMetricName("f1")

    //4.l Préparer la grid-search

    val GridSearchTrainSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    //lancer la grid-search sur le dataset “training” préparé précédemment.
    val CVModel = GridSearchTrainSplit.fit(df_training)

    //val new_df_training = CVModel.transform(df_training)



    // 5.m Appliquer le meilleur modèle trouvé avec la grid-search aux données test
    val df_WithPredictions = CVModel.transform(df_test)


    //5.n Afficher df_WithPredictions.groupBy("final_status", "predictions").count.show()

    df_WithPredictions.groupBy("final_status", "predictions").count.show()

    //5 Afficher le f1-score du modèle sur les données de test.
    val f1_score = evaluator.evaluate(df_WithPredictions).toString

    println("f1 score:  %s".format(f1_score))

    //5 Sauvegarder le modèle

    CVModel.write.overwrite().save(path = "/Users/tetianabovkun/spark-2.2.0-bin-hadoop2.7/prepared_trainingset/data/Model")




     // println("hello world ! from Trainer")


  }

}
