# Near-Earth Objects(NEO) Hazardous Classification
## Goal
The aim of this project is to predict whether an object passing in the vicinity of the earth should be treated as hazardous based on the dataset provided.

## Approach/ Design Choices
### a. Process data
- After analyzing the pair-plots, we have dropped the features which are either constant or do not add any significance to predictions namely - id, name, sentry object, and orbiting body, relative velocity.
- Features utilized - est_diameter_min, est_diameter_max, miss_distance, and absolute_magnitude

### b. Data transformation and handling outliers
- The data is normalized using the z-score table.
- We tried different methods to handle outliers like replacing with percentiles and medians before removing it completely.
- We realized removing outliers provides better performance than replacing them. Total of 2.94 % data was classified as outliers.

### c. Model selection
- After analyzing the data it was understood that the structure of data is non-linear.
- The Support Vector Machine model would perform well in this case since it uses kernel tricks to perform more efficiently.Hence, the SVM model is picked for the purpose of these predictions.
- After doing grid search and providing the C and gamma values, it was found that predictions did not change after tuning these parameters with the model.

### d. Principal Component Analysis
- The top 2 components returned by PCA are **miss distance** and **absolute magnitude.**
- The model predictions **remained the same** after performingPCA.

### e. Optimizations

- When we downsampled, the prediction of the case **FalsePositive** (Model predicted object is hazardous but it actually is not) increased from 0.2% to 11%. But the case of **True Negative** (Model predicted object is not hazardous but it actually is) reduced dramatically from 7% to 0.1%.

- So the downsampling actually is better since we do not want to falsely predict that object is **not** hazardous when it **actually is**.