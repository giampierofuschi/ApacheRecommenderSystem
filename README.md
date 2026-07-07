# Movie Recommender System
### Final Graduation Project - University of Palermo (UNIPA)

This project implements a **Movie Recommender System** based on the **ALS (Alternating Least Squares)** Machine Learning algorithm using the **Apache Spark** framework in Java. 

The application analyzes historical user movie ratings to predict user preferences and generate personalized recommendations (Top 5 movies per user and Top 5 users per movie).

> **Project Documentation:** The official, detailed graduation report (**Relazione**) is available directly inside this repository, written in Italian.

---

## Dataset
The dataset used for training and evaluating this model is sourced from the following repository:
* **Source:** [lynchblue/movie-rating-dataset](https://github.com/lynchblue/movie-rating-dataset)

The program expects the data file to be located at `data/ratings.csv` with the following columns:
* `id`: Unique identifier for the user (integer)
* `movie_id`: Unique identifier for the movie (integer)
* `rate`: Movie rating given by the user (float/decimal)
* `date`: Timestamp or date of the review (string)

---

## 🛠️ Tech Stack
* **Language:** Java 21
* **Distributed Computing Framework:** Apache Spark 3.5.0 (Modules: Spark Core, Spark SQL, Spark MLlib)
* **Dependency Management:** Maven
* **IDE:** IntelliJ IDEA

---
