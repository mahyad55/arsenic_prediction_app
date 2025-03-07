# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 12:36:05 2023

@author: 81150
"""

# Import Modules
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import inspect
import math
import logging
import re
import sys
import os
from decimal import Decimal

import traceback


#%% create log class
class Logger:

    def __init__(self, logLevel=logging.DEBUG, logFileName='my_log.log'):
        self.logLevel = logLevel
        self.logFileName = logFileName

    def my_logger(self):
        # set class/method name from where its called
        logger_name = inspect.stack()[1][3]

        # create logger
        logger = logging.getLogger(logger_name)

        # set logger level
        logger.setLevel(self.logLevel)

        # create file handler and set format
        file_handler = logging.FileHandler(self.logFileName)
        formatter = logging.Formatter('%(asctime)s,%(levelname)s,%(name)s,%(message)s')

        # add formatter to File Handler
        file_handler.setFormatter(formatter)

        # add File Handler to logger
        logger.addHandler(file_handler)

        return logger


#%% Error finder decorator
def error_finder(func):
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as error:
            # Get the traceback information
            _, _, tb = sys.exc_info()
            tb_info = traceback.extract_tb(tb)
            # Extract the information for the relevant frame
            error_frame = tb_info[-1]
            error_line = error_frame[1]
            file_name = error_frame[0]
            function_name = error_frame[2]
            error_type = type(error).__name__
            error_text = error
            runtime_error = f'file_name     : {file_name}\n' \
                            f'function_name : {function_name}\n' \
                            f'error_line    : {error_line}\n' \
                            f'error_type    : {error_type}\n' \
                            f'error_text    : {error_text}'
            print('=' * 79)
            print(runtime_error)
            print('=' * 79)

    return wrapped_func


#%% Error logger decorator
logger_instance = Logger()

def error_logger(my_logger_instance):
    def decorator(func):
        def wrapped_func(*args, **kwargs):
            try:
                # my_logger_instance.my_logger().info(f"{func.__name__} executed successfully.")
                return func(*args, **kwargs)
            except Exception as error:
                # Get the traceback information
                _, _, tb = sys.exc_info()
                tb_info = traceback.extract_tb(tb)
                # Extract the information for the relevant frame
                error_frame = tb_info[-1]
                error_line = error_frame[1]
                file_name = error_frame[0]
                function_name = error_frame[2]
                error_type = type(error).__name__
                error_text = error

                runtime_error = f'file_name     : {file_name}\n' \
                                f'function_name : {function_name}\n' \
                                f'error_line    : {error_line}\n' \
                                f'error_type    : {error_type}\n' \
                                f'error_text    : {error_text}'
                print('=' * 79)
                print(runtime_error)
                print('=' * 79)

                error_message = f'file_name: {file_name}, ' \
                                f'function_name: {function_name}, ' \
                                f'error_line: {error_line}, ' \
                                f'error_type: {error_type}, ' \
                                f'error_text: {error_text}'
                my_logger_instance.my_logger().error(error_message)
                # Re-raise the exception to propagate it further
                # raise error

        return wrapped_func

    return decorator

# %% HumanReadableNumber
def HumanReadableNumber(n):
    millnames = ['','K','M','B','T']

    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))

    return '{:.1f}{}'.format(n / 10**(3 * millidx), millnames[millidx])


#%% Date Utilities
class DateUtils:

    # Convert gregorian to jalali
    # @error_logger(logger_instance)
    @error_finder
    def gregorian_to_jalali(self, year, month, day, splitter='-'):
        g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

        jy = 0 if year <= 1600 else 979
        year -= 621 if year <= 1600 else 1600
        year2 = year + 1 if month > 2 else year
        days = (365 * year) + int((year2 + 3) / 4) - int((year2 + 99) / 100)
        days += int((year2 + 399) / 400) - 80 + day + g_d_m[month - 1]
        jy += 33 * int(days / 12053)
        days %= 12053
        jy += 4 * int(days / 1461)
        days %= 1461
        jy += int((days - 1) / 365)

        if days > 365:
            days = (days - 1) % 365

        if days < 186:
            jm = 1 + int(days / 31)
            jd = 1 + (days % 31)
        else:
            arit = days - 186
            jm = 7 + int(arit / 30)
            jd = 1 + (arit % 30)

        if jm < 10:
            jm = '0' + str(jm)

        if jd < 10:
            jd = '0' + str(jd)

        y = str(jy) + splitter + str(jm) + splitter + str(jd)
        return y

    # Convert Jalali datetime to Gregorian datetime
    # @error_logger(logger_instance)
    @error_finder
    def jalali_to_gregorian(self, year, month, day, splitter='-'):

        jy = int(year)
        jm = int(month)
        jd = int(day)
        jy += 1595
        days = -355668 + (365 * jy) + ((jy // 33) * 8) + (((jy % 33) + 3) // 4) + jd
        if jm < 7:
            days += (jm - 1) * 31
        else:
            days += ((jm - 7) * 30) + 186
        gy = 400 * (days // 146097)
        days %= 146097
        if days > 36524:
            days -= 1
            gy += 100 * (days // 36524)
            days %= 36524
            if days >= 365:
                days += 1
        gy += 4 * (days // 1461)
        days %= 1461

        if days > 365:
            gy += ((days - 1) // 365)
            days = (days - 1) % 365
        gd = days + 1
        if (gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0):
            kab = 29
        else:
            kab = 28
        sal_a = [0, 31, kab, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        gm = 0
        while gm < 13 and gd > sal_a[gm]:
            gd -= sal_a[gm]
            gm += 1

        if gm < 10:
            gm = '0' + str(gm)

        if gd < 10:
            gd = '0' + str(gd)

        y = str(gy) + splitter + str(gm) + splitter + str(gd)
        y = pd.to_datetime(y)

        return y

    # @error_logger(logger_instance)
    @error_finder
    def date_diff(self, date_of_validity, num_months, num_days=0):
        new_date = pd.to_datetime(date_of_validity) - \
                   relativedelta(months=num_months) - \
                   relativedelta(days=num_days)

        return new_date

    @error_finder
    # Get Date of Last/Nest N Months date
    def get_date(self, date_of_validity, num_months, direction, **kwargs):

        if 'num_days' in kwargs:
            num_days = kwargs.get('num_days')
        else:
            num_days = 0

        direction = direction.lower()
        if direction=='backward':
            num_months = -abs(num_months)
            num_days = -abs(num_days)
        elif direction=='forward':
            num_months = abs(num_months)
            num_days = abs(num_days)

        if 'input_format' in kwargs:
            input_format = kwargs.get('input_format')
        else:
            year  = str(date_of_validity)[:4]
            if int(year)<1500:
                input_format = 'jalali'
            else:
                input_format = 'gregorian'

        if input_format=='jalali':
            date_of_validity = str(date_of_validity)
            date_of_validity = re.sub(r'[^0-9]','',date_of_validity)
            if len(date_of_validity)>=8:
                year  = date_of_validity[:4]
                month = date_of_validity[4:6]
                day   = date_of_validity[6:8]

                date_of_validity = self.jalali_to_gregorian(year,month,day)
            else:
                raise SyntaxError(
                    'data_of_validiy should have format like this: "yyyy-mm-dd"'
                )

        return self.date_diff(date_of_validity, num_months, num_days)

    # calculate number of years between two datetime
    # @error_logger(logger_instance)
    @error_finder
    def number_of_years(self, d1, d2):
        return relativedelta(d1, d2).years

    # calculate number of months between two datetime
    # @error_logger(logger_instance)
    @error_finder
    def number_of_months(self, d1, d2):
        return relativedelta(d1, d2).months + \
               (relativedelta(d1, d2).years * 12)

    # calculate number of days between two datetime
    # @error_logger(logger_instance)
    @error_finder
    def number_of_days(self, d1, d2):
        return (d1 - d2).days


# %% numerize
def round_num(n, decimal=2):
    n=Decimal(n)
    return n.to_integral() if n == n.to_integral() else round(n.normalize(), decimal)

def numerize(n, decimal=2):
    #60 sufixes
    sufixes = [ "", "K", "M", "B", "T", "Qa", "Qu", "S", "Oc", "No",
                "D", "Ud", "Dd", "Td", "Qt", "Qi", "Se", "Od", "Nd","V",
                "Uv", "Dv", "Tv", "Qv", "Qx", "Sx", "Ox", "Nx", "Tn", "Qa",
                "Qu", "S", "Oc", "No", "D", "Ud", "Dd", "Td", "Qt", "Qi",
                "Se", "Od", "Nd", "V", "Uv", "Dv", "Tv", "Qv", "Qx", "Sx",
                "Ox", "Nx", "Tn", "x", "xx", "xxx", "X", "XX", "XXX", "END"]

    sci_expr = [1e0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24, 1e27,
                1e30, 1e33, 1e36, 1e39, 1e42, 1e45, 1e48, 1e51, 1e54, 1e57,
                1e60, 1e63, 1e66, 1e69, 1e72, 1e75, 1e78, 1e81, 1e84, 1e87,
                1e90, 1e93, 1e96, 1e99, 1e102, 1e105, 1e108, 1e111, 1e114, 1e117,
                1e120, 1e123, 1e126, 1e129, 1e132, 1e135, 1e138, 1e141, 1e144, 1e147,
                1e150, 1e153, 1e156, 1e159, 1e162, 1e165, 1e168, 1e171, 1e174, 1e177]
    minus_buff = n
    n=abs(n)
    for x in range(len(sci_expr)):
        try:
            if n >= sci_expr[x] and n < sci_expr[x+1]:
                sufix = sufixes[x]
                if n >= 1e3:
                    num = str(round_num(n/sci_expr[x], decimal))
                else:
                    num = str(n)
                return num + sufix if minus_buff > 0 else "-" + num + sufix
        except IndexError:
            print("You've reached the end")


def denumerize(n):
    try:

        sufixes = [ "", "K", "M", "B", "T", "Qa", "Qu", "S", "Oc", "No",
                    "D", "Ud", "Dd", "Td", "Qt", "Qi", "Se", "Od", "Nd","V",
                    "Uv", "Dv", "Tv", "Qv", "Qx", "Sx", "Ox", "Nx", "Tn", "Qa",
                    "Qu", "S", "Oc", "No", "D", "Ud", "Dd", "Td", "Qt", "Qi",
                    "Se", "Od", "Nd", "V", "Uv", "Dv", "Tv", "Qv", "Qx", "Sx",
                    "Ox", "Nx", "Tn", "x", "xx", "xxx", "X", "XX", "XXX", "END"]

        sci_expr = [1e0, 1e3, 1e6, 1e9, 1e12, 1e15, 1e18, 1e21, 1e24, 1e27,
                      1e30, 1e33, 1e36, 1e39, 1e42, 1e45, 1e48, 1e51, 1e54, 1e57,
                      1e60, 1e63, 1e66, 1e69, 1e72, 1e75, 1e78, 1e81, 1e84, 1e87,
                        1e90, 1e93, 1e96, 1e99, 1e102, 1e105, 1e108, 1e111, 1e114, 1e117,
                      1e120, 1e123, 1e126, 1e129, 1e132, 1e135, 1e138, 1e141, 1e144, 1e147,
                        1e150, 1e153, 1e156, 1e159, 1e162, 1e165, 1e168, 1e171, 1e174, 1e177]

        suffix_idx = 0

        for i in n:
              if type(i) != int:
                  suffix_idx = n.index(i)


        suffix = n[suffix_idx:]
        num = n[:suffix_idx]

        suffix_val = sci_expr[sufixes.index(suffix)]

        result = int(num) * suffix_val

        return result

    except:
        return 'err'