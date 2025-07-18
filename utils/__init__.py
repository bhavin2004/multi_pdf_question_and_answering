import logging
from datetime import datetime
import os
import sys

LOG_FILE=f"{datetime.now().strftime('%d-%m-%Y-%H_%M_%S')}.log"

LOG_PATH = os.path.join(os.getcwd(),'logs',LOG_FILE)

LOG_FILE_PATH = os.path.join(LOG_PATH,LOG_FILE)
os.makedirs(LOG_PATH,exist_ok=True)


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]: %(levelname)s: %(name)s %(module)s at line no %(lineno)d=> %(message)s",
    handlers=[
        logging.FileHandler(filename=LOG_FILE_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()


def get_error_details(error_msg: Exception, error_detail) -> str:
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename

    error_message = f"The error Occurred in Python Script {filename} at line no:{exc_tb.tb_lineno}\nAnd Error:{error_msg}"

    return error_message


class CustomException(Exception):
    def __init__(self, error: Exception, error_detail):
        super().__init__(error)
        self.error_msg = get_error_details(error, error_detail)
        logger.error(self.error_msg)

    def __str__(self):
        return self.error_msg


if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        raise CustomException(e, sys)
