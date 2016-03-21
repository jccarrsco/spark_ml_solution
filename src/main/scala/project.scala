import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
/*
 * Written by: Jose Carlos Carrasco Jimenez
 * Purpose: Classify documents based on vocabulary in common
*/

object Project extends App {
	val spConfig = (new SparkConf)
							.setMaster("local")
							.setAppName("SparkMLProject")

		val sc = new SparkContext(spConfig)

		// Load the data of the form (DocumentID, WordID, Count) -> how many times wordID appears in DocumentID
		// Properties of the data (39,861 documents --- 28,099 words in vocabulary --- 3,710,420 words in the collection)
		val dataFile = "src/main/resources/data/docword.enron.txt"
		val data = sc.textFile(dataFile)

		// Parse data from String to (DocumentID, WordID, Count)
		val parsedData = data.map(l => l.split(" "))
							.map{case Array(docID, wordID, count) => (docID.toInt, wordID.toInt, count.toDouble)}
							.cache()

		// Get unique words in order to build the feature space
		// Features are the words in the vocabulary. In other words, we will have 28099 features describing each documents
		val features = parsedData.map{case (docID, wordID, count) => wordID}.distinct
		val numFeatures = features.count.toInt

		// Create a map from wordID to an index
		val featureMap = features.zipWithIndex.map{case (value, index) => (value, index.toInt)}
											  .collectAsMap

		// We need to transform from (DocumentID, WordID, Count) to RDD(docID, wordID1, wordID2, ..., wordIDN)
		// First, we groupBy documentID
		val docWithListOfWords = parsedData.map{case (docID, wordID, count) => (docID, (wordID, count))}	// convert to a 2-Tuple
										.groupByKey()	// We have an RDD of the form [docID, CompactBuffer( (wordID1, 14), (wordID2, 5), ...)]

		val docIDs = docWithListOfWords.map{case (docID, pairs) => docID}

		//val documents = docWithListOfWords.map{case (docID, words) => Vectors.sparse( numFeatures.toInt,
		//  									words.map{case (wordID, count) => featureMap.getOrElse(wordID, 0)}.asInstanceOf[Array[Int]],
		//										words.map{case (wordID, count) => count.toDouble}.asInstanceOf[Array[Double]] )}

		val documents = docWithListOfWords.map{case (docID, words) => Vectors.sparse(numFeatures,
										words.map{case (wordID, count) => (featureMap.getOrElse(wordID, -1), count ) }.toSeq  )}


		// define parameters for clustering algorithm
		val numClusters = 50
		val numIterations = 10

		// data segmentation process
		val model = KMeans.train(documents, numClusters, numIterations)

		// Evaluate model using Within Sum of Squared Errors
		val WSSE = model.computeCost(documents)

		// Display WSSE
		println("WSSE: " + WSSE + "\n")

		// Obtain the group to which each document belongs
		val clusters = model.predict(documents)

		// Create a list of pairs of the type (docID, cluster)
		val documentGroups = docIDs.collect().zip(clusters.collect())

		// display first 50 documents and their groups
		documentGroups.take(50).foreach(println)


		// spark-submit --class "Project" --master local[4] target/scala-2.10/sparkmlproject_2.10-1.0.jar

		// Save model
		//val modelFile = ""
		//clusters.save(sc, modelFile)
}
