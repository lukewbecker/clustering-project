# Zillow Regression Project

### Author: Luke Becker

## Description: 
The purpose of this project is to develop a model that is able to predict the log error of Zillow's Zestimate in predicting values in three counties in California, using the Zillow dataset.

In addition, there are these deliverables:
    1. Acquire.py, Prep.py, Wrangle.py, and explore.py files to assist in recreating project.
    2. Juptyer Notebook with exploration and modeling of the data.
    3. Readme with summary of project outline and data dictionary.

## Project Goals

- Discover if by using clustering models and algorithmns, I can produce a model that better predicts the log error of Zillow's Zestimate of property values.
- Utilize Clustering techniques to produce visuals and new features for modeling


## Project Planning

Initial Questions:
- Does number of bedrooms matter to the log error of the Zestimate?
- Does location within the state or county affect overall log error?
- Are there clusters of related features, such as location information that can yield more useful derived features to model on?
- Do these features created from clustering actually produce a better model to predict the Zestimate's log error?



### Hypothesis Tests:

Initial ideas for testing.

##### Features: `latitude` and `logitude`

- H0: The population means for the 4 location clusters are all equal (**no** significant difference)

- Ha: The population means for the 4 location clusters are not equal (there **is** a significant difference)

##### Features: bedroomcnt and bathroomct

- H0: The population means for the 4 rooms clusters are all equal (**no** significant difference)

- Ha: The population means for the 4 rooms clusters are not equal (there **is** a significant difference)


## Data Dictionary

| Feature | Definition |
| --- | --- |
| bathroomcnt | Number of bathrooms in property (includes half bathrooms) |
| bedroomcnt | Number of bedrooms in property |
| calculatedbathnbr | Number of both bedrooms and bathrooms in property |
| calculatedfinishedsquarefeet | Total Square Footage of the property |
| fullbathcnt | Number of full bathrooms in property (excludes half bathrooms) |
| centroid_latitude | The centroid of latitude created via clustering |
| centroid_longitude | The centroid of longitutde created via clustering |
| age | Number of years since house was built |
| location_cluster_0 | Subset of the location clustering | 
| fips | A federal code designating areas of the country that functions similarly to a zip code |

| Target | Definition |
| --- | --- |
| logerror | Log error of Zestimate vs actual property value |
