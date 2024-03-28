## Description

A strawberry grower is having difficulty forecasting their weekly production quantities. An engineering team has provided me with a dataset, and my mission is to analyze this data to create value for the farmers. This includes offering a predictive model or strategy for more accurate harvest forecasting.
Initially, we have data from images taken by cameras in greenhouses, data from sensors, as well as harvest data. The data from the images represents the parameters of fruits or flowers, which are updated daily. Data from the sensors is updated every hour and describes external conditions, such as air temperature during the day, carbon dioxide content, etc. The third input file contains the results of a harvest that did not occur every day.
Data from the images helps us understand when the phases of fruit ripening change. I evaluated the influence of each factor on a specific phase. The research includes the following phases:

 - *Flower*: This is the initial phase when the strawberry plant develops flowers.
 - *Fruitset*: In this phase, the flowers begin to form fruits.
 - *Smallgreen*: The fruits begin to grow and develop, but they remain green and small.
 - *Turning*: The berries begin to change color from green to red, indicating the beginning of the ripening process.
 - *White (maturity)*: Berries reach ripeness, usually becoming brighter in color and becoming juicier and sweeter.
 - *Red (full ripeness)*: This is the phase when the berries are fully ripe and ready to be picked.

For the final results, we need information from the file containing the harvest results. The target variable of early phases is the number of berries multiplied by their size. The target variable of the last phase is the mass of berries, since I need to predict the mass of the harvest weight.
The sensor data contains some gaps that can be eliminated using certain methods, for example, from the Pandas library. To develop the prediction model, I used the average daily temperature, maximum and minimum daily temperature. This part of the work can be further improved, since we can take into account how long the temperature remained high during the day and low at night. However, this would take a significant amount of time and effort and since I had limited time, I decided to use simpler features that allowed me to build high-quality predictive models.
For more convenient work with the data I grouped some of them by day.

So, according to my idea, we need to separate each phase and evaluate the influence of various factors on these phases. I assumed that the current phase is influenced by the same phase, as well as the previous one. To separate the phases I did the following. I used the duration of each phase and aligned the start of the first phase with the start of the second phase. In this case, the start date of the phase is not saved, however, this is not important. The important thing is that we can now look at the two phases synchronously. Of course, the flowering phase does not end at any particular moment, since flowers continue to appear until the end of the measurements, we do not need to think about the duration of each phase being different. So, in order to do this, I shifted each phase by a certain amount of time.
Another solution at this stage is to combine all the data into one DataFrame, combine the beginning of all phases and build a model that takes into account the influence of all phases on the final result. This would help to understand which phase is most important. However, according to my assumption, each phase affects only the next one, and I neglect the influence on later phases.
In my solution, the flower phase only has its own data, since it is the first phase. The last phase (red) is merged with the final DataFrame, which contains harvest data.
Three models were used to solve the problem: XGBoost, RandomForest and a linear model. Of course, they are radically different from each other. The linear model takes into account only linear dependencies between features, RandomForest tries to minimize cross-influence and overfitting, and XGBoost is also built on the work of trees and uses gradient descent to quickly find the global minimum.
The graphs in the Modeling.ipynb file show the predictions of all three models and the actual data. At first, with very little data, the models perform poorly, but then their performance improves.
It would be valuable to accumulate more data over multiple runs so that the models can learn better and perform more accurately.
Along with the graphs, you can look at the model quality metrics – RMSE and MAPE.

The solution contains three Python scripts, as well as two Jupyter notebooks, which use the above-mentioned Python scripts. The scripts were placed in separate files for better code readability.
Two of the Python scripts, datapreprocessing.py and utils.py, are used to prepare the data for work so that new data that will arrive in the future can also undergo this processing, which we use to test and find an approach. Another Python script train_model.py, is used to build predictive models. 
The requirements.txt file contains all the necessary libraries to run the files for this solution. To install them all at once, just run the pip install -r requirements.txt command on the command line.
