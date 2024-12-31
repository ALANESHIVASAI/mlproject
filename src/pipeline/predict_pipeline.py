import sys
import os
import pandas as pd
from src.utils import loadobj
from src.exception import CustomException
from src.logger import logging

class predictpipeline:

    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path="artifacts\model.pkl"
            preprocessor_path='artifacts\preprocessor.pkl'

            model=loadobj(file_path=model_path)
            preprocessor=loadobj(file_path=preprocessor_path)

            logging.info(f"{ preprocessor }")

            data=preprocessor.transform(features)
            pred=model.predict(data)

            return pred
        except Exception as e:
            raise CustomException(e,sys)


class customData:
    def __init__(
            self,
            gender,
            ethnicity,
            parental_level_of_education,
            lunch,
            test_preparation_course,
            reading_score,
            writing_score
                ):
        
        self.gender = gender

        self.race_ethnicity = ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
    