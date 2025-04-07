import sys
import os
import logging
import pandas as pd

import ollama
import json

import collections.abc
import re
from io import StringIO

from acl_py_util import acl_py_util
from acl_py_util import logger

def extract_result_to_dataframe(result_string):
    """
    Extracts the content between <result></result> tags and converts it into a pandas DataFrame.

    Parameters:
    - result_string (str): The input string containing the <result></result> block.

    Returns:
    - pd.DataFrame: A DataFrame containing the extracted data.
    """
    # Extract content between <result> and </result>
    match = re.search(r"<result>(.*?)</result>", result_string, re.DOTALL)
    if not match:
        raise ValueError("No <result> block found in the input string.")
    
    result_content = match.group(1).strip()

    # Convert the content to a DataFrame
    data = StringIO(result_content)
    df = pd.read_csv(data, skipinitialspace=True)
    
    return df



def main(args):
    error_file = os.getenv("ACL_PY_ERROR_FILE")
    if not error_file:
        error_file = os.getenv('LOCALAPPDATA') + "/acl_py_utl.error"

    df = acl_py_util.from_an()

    logger.info(df)

    #sample user code df
    data_str = df.to_string(index=False)
    user_request = "duplicate payments"
    columns_to_check = ' '.join(df.columns)
    column_names = ' '.join(df.columns.str.strip())
    columns_inresult = 'last_name, first_name, pay_on, salary, record_number'
 #   columns_to_check = "last_name and first_name and salary and pay_on must be the same"

    sys_prompt = f"""
    You must only answer factually and concisely. If unsure, say 'I don't know'.
    Ensure the result is enclosed within the <result></result> tags.
    Return the CSV data within the <result></result> tags.
    """

    user_prompt = f"""
    Analyze the following data and find all {user_request} based on {columns_to_check}. 
    Return the result data strictly as a CSV format containing the columns: {columns_inresult}. 
 
    {data_str}
    """

    # Send the request to Ollama
    response = ollama.chat(model="mistral", messages=[{"role": "system", "content": sys_prompt},
                                                      {"role": "user", "content": user_prompt}], options={"temperature": 0.0})
    print("response")
    logger.info(response["message"]["content"])   
    try:
        usrdf = extract_result_to_dataframe(response["message"]["content"])
    except ValueError:
        logger.error("ZeroDivisionError: No <result> block found in the input string.")
        return -1

    print(usrdf)   
    #end user operations


    acl_py_util.to_an(usrdf)

    logger.info(f"done")
  
if __name__ == "__main__":
    main(sys.argv)