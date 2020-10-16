# Zillow Regression Project

### Authors: Austin Aranda, Luke Becker

## Description: 
The purpose of this project is to develop a model that is able to predict property values in Los Angeles, California, using the Zillow dataset. We intend to discover the key features that most accurately predict property value. 

In addition, we plan on these deliverables:
    1. Map showing variations in tax rates by county.
    2. Acquire.py, Prep.py, Wrangle.py, and Model.py files to be able to recreate our efforts.
    3. Presentation with the results of our findings.

## Project Goals



## Project Planning

Initial Questions:
- Does number of bedrooms matter to the property value?
- Does location within the state or county affect overal property value?
- Is total square footage independent of property value?
- Does a combined bedroom/bathroom square footage feature have better correlation with the target variable than the separate features?


### Hypotheses:

- $ H_0 $: Properties with 2 or more bedrooms do not have higher value than average property

- Ha: Properties with 2 or more bedrooms do have higher value than average property

- H0: Properties with 2 or more bedrooms do not have more square footage than average property

- Ha: Properties with 2 or more bedrooms do not have more square footage than average property


## Data Dictionary

| Feature | Definition |
| --- | --- |
| bathroomcnt | Number of bathrooms in property (includes half bathrooms) |
| bedroomcnt | Number of bedrooms in property |
| calculatedbathnbr | Number of both bedrooms and bathrooms in property |
| calculatedfinishedsquarefeet | Total Square Footage of the property |
| fullbathcnt | Number of full bathrooms in property (excludes half bathrooms) |

| Target | Definition |
| --- | --- |
| taxvaluedollarcnt | Value of the property |
