import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,err_tbr=error_detail.exc_info()
    file_name=err_tbr.tb_frame.f_code.co_filename
    error_message="Error occured in Python Script filename[{0}] in line [{1}] error message [{2}]".format(file_name,err_tbr.tb_lineno,str(error)
          )
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail=error_detail)

    def __str__(self):
        return self.error_message
    