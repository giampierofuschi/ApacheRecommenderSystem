package org.example;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Encoders;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;

import java.io.Serializable;

import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;

public class Main {

    public static class Rating implements Serializable {
        private int id;
        private int movie_id;
        private float rate;
        private String date;

        public Rating() {}

        public int getId() { return id; }
        public void setId(int id) { this.id = id; }

        public int getMovie_id() { return movie_id; }
        public void setMovie_id(int movie_id) { this.movie_id = movie_id; }

        public float getRate() { return rate; }
        public void setRate(float rate) { this.rate = rate; }

        public String getDate() { return date; }
        public void setDate(String date) { this.date = date; }
    }

    public static void main(String[] args) {

        SparkSession spark = SparkSession.builder()
                .appName("CSVRecommenderSystem")
                .master("local[*]")
                .getOrCreate();

        // log level impostato su error così da nascondere info e warn
        spark.sparkContext().setLogLevel("ERROR");

        // vengono caricati dentro ratingsRaw le righe di ratings.csv
        String path = "data/ratings.csv";

        Dataset<Row> ratingsRaw = spark.read()
                .option("header", "true")   // la prima riga del csv viene considerata header
                .option("inferSchema", "true") // riconosce i tipi automaticamente (int, float...)
                .csv(path);

        // ogni riga del csv viene convertita in un oggetto della classe rating
        Dataset<Rating> ratingsDS = ratingsRaw.as(Encoders.bean(Rating.class));
        Dataset<Row> ratings = ratingsDS.toDF(); // converte in dataframe, più adatto per spark

        // si prepara training set (80%) e test set (20%)
        Dataset<Row>[] splits = ratings.randomSplit(new double[]{0.8, 0.2});
        Dataset<Row> training = splits[0];
        Dataset<Row> test = splits[1];

        // viene applicato l'algoritmo ALS sul training set
        ALS als = new ALS()
                .setMaxIter(10)      // 10 iterazioni, corrisponde a valore di default
                .setRegParam(0.1)    // valore lambda per evitare overfitting
                .setUserCol("id")
                .setItemCol("movie_id")
                .setRatingCol("rate");

        ALSModel model = als.fit(training);

        // viene valutato il modello sul test set
        model.setColdStartStrategy("drop");
        Dataset<Row> predictions = model.transform(test);

        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setMetricName("rmse")
                .setLabelCol("rate")
                .setPredictionCol("prediction");

        double rmse = evaluator.evaluate(predictions);
        System.out.println("Precisione (Root Mean Squared Error) = " + rmse);

        // viene generata la lista di raccomandazioni per ogni utente
        System.out.println("Top 5 film per ogni utente:");
        Dataset<Row> userRecs = model.recommendForAllUsers(5);
        userRecs.show(5, false);

        // viene generata la lista di raccomandazioni per ogni item
        System.out.println("Top 5 utenti per ogni film:");
        Dataset<Row> movieRecs = model.recommendForAllItems(5);
        movieRecs.show(5, false);

        spark.stop();
    }
}