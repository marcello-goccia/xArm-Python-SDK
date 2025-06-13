import sys
import traceback

# import logging
# logger = logging.getLogger(__name__)


def get_debug_info():
   """
   This method returns the string with the information of what caused the exception to be raised.

   @return string the value with the debug info to write on the log file
   """
   string = f" {sys.exc_info()[0]}: "
   for frame in traceback.extract_tb(sys.exc_info()[2]):
       file_name, line_no, function, text = frame
       if file_name is None:
           file_name = ''
       if line_no is None:
           line_no = ''
       if function is None:
           function = ''
       if text is None:
           text = ''
       string += f" in file: {file_name}, line no: {line_no}, function: {function}, text: {text}\n"
   return string
