import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{VectorAssembler,StringIndexer,VectorIndexer,OneHotEncoder}

object Clustering extends App {

  /////////////////////////////////
  // K MEANS PROJECT EXERCISE ////
  ///////////////////////////////

  // Source of the Data
  //http://archive.ics.uci.edu/ml/datasets/Wholesale+customers

  // Here is the info on the data:
  // 1)	FRESH: annual spending (m.u.) on fresh products (Continuous);
  // 2)	MILK: annual spending (m.u.) on milk products (Continuous);
  // 3)	GROCERY: annual spending (m.u.)on grocery products (Continuous);
  // 4)	FROZEN: annual spending (m.u.)on frozen products (Continuous)
  // 5)	DETERGENTS_PAPER: annual spending (m.u.) on detergents and paper products (Continuous)
  // 6)	DELICATESSEN: annual spending (m.u.)on and delicatessen products (Continuous);
  // 7)	CHANNEL: customers Channel - Horeca (Hotel/Restaurant/Cafe) or Retail channel (Nominal)
  // 8)	REGION: customers Region- Lisnon, Oporto or Other (Nominal)


  import org.apache.log4j._
  Logger.getLogger("org").setLevel(Level.ERROR)


  // Create Session App
  val spark = SparkSession.builder().master("local")
    .appName("ClusteringExample")
    .getOrCreate()

  // Use Spark to read in the csv file.
  val data = spark.read.option("header","true").option("inferSchema","true")
    .format("com.databricks.spark.csv").load("src/main/resources/wholesale_customers_data.csv")


  // Print the Schema of the DataFrame
  data.printSchema()


  // This is needed to use the $-notation
  import spark.implicits._
  val clustData = data.select($"Fresh",$"Milk", $"Grocery", $"Frozen",
    $"Detergents_Paper", $"Delicassen")

  // Import VectorAssembler and Vectors
  val assembler = new VectorAssembler()
    .setInputCols(Array("Fresh","Milk", "Grocery", "Frozen",
      "Detergents_Paper", "Delicassen"))
    .setOutputCol("features")


  // Use the assembler object to transform the feature_data
  val training_data = assembler.transform(clustData).select("features")

  // Create a Kmeans Model with K=3
  val kmeans = new KMeans().setK(3).setSeed(12345L)

  // Fit that model to the training_data
  val model = kmeans.fit(training_data)

  // Evaluate clustering by computing Within Set Sum of Squared Errors.
  val WSSSE = model.computeCost(training_data)
  println(s"Within Set Sum of Squared Errors = $WSSSE")


  // Shows the result.
  println("Cluster Centers: ")
  model.clusterCenters.foreach(println)

}
