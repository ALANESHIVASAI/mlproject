import sys
import os
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts",'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.initiate_transformation_config=DataTransformationConfig()

    def get_transformer_obj(self):
        try:
            numeric_features=['reading_score','writing_score']
            categorical_features=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info("Numerical features encoding completed")
            categorical_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            logging.info("Categorical features encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numeric_features),
                    ("cat_pipeline",categorical_pipeline,categorical_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read the Train and test data")

            logging.info("Obtaining the preprocessor Object")

            preprocessor_obj=self.get_transformer_obj()

            target_column="math_score"

            input_feature_train_df=train_df.drop(columns=[target_column],axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=[target_column],axis=1)
            target_feature_test_df=test_df[target_column]

            logging.info("Applying the transformation on the the train and test data")

            input_feature_train=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train,np.array(target_feature_train_df)]

            test_arr=np.c_[input_feature_test,np.array(target_feature_test_df)]

            save_object(
                file_path=self.initiate_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info("saved the .pkl file")
            
            return(
                train_arr,
                test_arr,
                self.initiate_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)

