# Zillow Regression Project

### Authors: Austin Aranda, Luke Becker

## Description: 
The purpose of this project is to develop a model that is able to predict property values in Los Angeles, California, using the Zillow dataset. We intend to discover the key features that most accurately predict property value. 

In addition, we plan on these deliverables:
    1. Map showing variations in tax rates by county.
    2. Acquire.py, Prep.py, Wrangle.py, and Model.py files to be able to recreate our efforts.
    3. Presentation with the results of our findings.

## Project Organization

Generated with [ryans_codeup_data_science_mvp](https://github.com/RyanMcCall/ryans_codeup_data_science_mvp)

Modified from [datasciencemvp](https://github.com/cliffclive/datasciencemvp/)

```
├── README.md               <- The top-level README for developers using this project.
│
├── data                    <- All of the data for the project
│   ├── modeling            <- The prepared, processed and split datasets for modeling.
│   ├── prepared            <- The prepared datasets for exploration
│   └── raw                 <- The original, immutable data
│
├── main.py                 <- The main python script that calls all src scripts
│
├── mvp.ipynb               <- The main notebook for the project
│
├── src                     <- The source code for use in this project
│   ├── __init__.py         <- Makes src a Python module
│   ├── acquire.py          <- The script to download or generate data and store it in
│   │                          data/raw/
│   ├── explore.py          <- The script for creating any visuals that need to be stored
│   │                          in visuals/generated_graphics/
│   ├── model.py            <- The script for preprocessing, modeling, and interpreting
│   └── prepare.py          <- The script for preparing the raw data and storing it in
│                              data/prepared/
│
└── visuals                 <- All project visuals
    ├── external_visuals    <- Visuals brought from outside the project
    ├── generated_graphics  <- Visuals generated from the project
    └── presentation        <- A copy of your presentation
```

## Project Planning

Initial Questions:
- Does number of bedrooms matter to the property value?
- Does location within the state or county affect overal property value?
- Is total square footage independent of property value?
- Does a combined bedroom/bathroom square footage feature have better correlation with the target variable than the separate features?


### Hypotheses:

- H0: Properties with 2 or more bedrooms do not have higher value than average property

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
