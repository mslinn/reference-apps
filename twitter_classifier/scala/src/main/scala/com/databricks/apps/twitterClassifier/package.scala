package com.databricks.apps

package twitterClassifier {
  import org.apache.spark.SparkContext
  import org.apache.spark.sql.{SQLContext, SparkSession}

  object SparkSetup {
    import org.apache.spark.SparkConf
    val maybeMaster: Option[String] = new SparkConf().getOption("spark.master")

    val spark: SparkSession =
      if (maybeMaster.isDefined) {
        println(s"Using master (${ maybeMaster.mkString }) specified on the command line.")
        SparkSession
          .builder
          .appName(getClass.getSimpleName.replace("$", ""))
          .getOrCreate()
      } else {
        println("No master was specified on the command line so running locally using all cores.")
        SparkSession
          .builder
          .master("local[*]")
          .appName(getClass.getSimpleName.replace("$", ""))
          .getOrCreate()
      }

      val sqlContext: SQLContext = spark.sqlContext

      val sc: SparkContext = spark.sparkContext
      // Suppress "WARN BlockManager: Block input-0-1478266015800 replicated to only 0 peer(s) instead of 1 peers" messages
      sc.setLogLevel("ERROR")
  }
}

package object twitterClassifier {
  import org.apache.spark.mllib.linalg.Vector
  import org.apache.spark.mllib.feature.HashingTF
  import twitter4j.auth.OAuthAuthorization
  import twitter4j.conf.ConfigurationBuilder

  val numFeatures = 1000
  val tf = new HashingTF(numFeatures)

  def maybeTwitterAuth: Some[OAuthAuthorization] = Some(new OAuthAuthorization(new ConfigurationBuilder().build))

  /** Create feature vectors by turning each tweet into bigrams of characters (an n-gram model)
    * and then hashing those to a length-1000 feature vector that we can pass to MLlib.
    * This is a common way to decrease the number of features in a model while still getting excellent accuracy
    * (otherwise every pair of Unicode characters would potentially be a feature). */
  def featurize(s: String): Vector = tf.transform(s.sliding(2).toSeq)
}
