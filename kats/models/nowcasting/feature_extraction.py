# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""This is a file with functions which turn time series into ML features.

Typical use case is to create various features for the nowcasting model.
The features are rolling, i.e. they are the times series as well.

  Typical usage example:

  >>> df = ROC(df, 5)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

def ROC(df: pd.DataFrame, n: int, column: str = 'y') -> pd.DataFrame:
    """Adds another column indicating return comparing to step n back.

    Args:
        df: a pandas dataframe.
        n: an integer on how many steps looking back.
        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added column.
    """

    M = df[column].diff(n - 1)
    N = df[column].shift(n - 1)
    if column == 'y':
        ROC = pd.Series(M / N, name=f'ROC_{n}')
    else:
        ROC = pd.Series(M / N, name=f'{column}_ROC_{n}')
    df = df.join(ROC)
    return df

def MOM(df: pd.DataFrame, n: int, column: str = 'y') -> pd.DataFrame:
    """Adds another column indicating momentum: difference of current value and n steps back.

    Args:
        df: a pandas dataframe.
        n: an integer on how many steps looking back.
        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added column.
    """

    if column == 'y':
        M = pd.Series(df[column].diff(n), name=f'MOM_{n}')
    else:
        M = pd.Series(df[column].diff(n), name=f'{column}_MOM_{n}')
    df = df.join(M)
    return df

def MA(df: pd.DataFrame, n: int, column: str = 'y') -> pd.DataFrame:
    """Adds another column indicating moving average in the past n steps.

    Args:
        df: a pandas dataframe.
        n: an integer on how many steps looking back.
        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added column.
    """

    if column == 'y':
        MA = pd.Series(df[column].rolling(n).mean(), name=f'MA_{n}')
    else:
        MA = pd.Series(df[column].rolling(n).mean(), name=f'{column}_MA_{n}')
    df = df.join(MA)
    return df

def LAG(df: pd.DataFrame, n: int, column: str = 'y') -> pd.DataFrame:
    """Adds another column indicating lagged value at the past n steps.

    Args:
        df: a pandas dataframe.
        n: an integer on how many steps looking back.
        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added column.
    """

    N = df[column].shift(n)
    if column == 'y':
        LAG = pd.Series(N, name=f'LAG_{n}')
    else:
        LAG = pd.Series(N, name=f'{column}_LAG_{n}')
    df = df.join(LAG)
    return df

def MACD(df: pd.DataFrame, n_fast: int =12, n_slow: int =21, column: str = 'y') -> pd.DataFrame:
    """Adds three columns indicating MACD: https://www.investopedia.com/terms/m/macd.asp.

    Args:
        df: a pandas dataframe
        n_fast: an integer on how many steps looking back fast.
        n_slow: an integer on how many steps looking back slow.
        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added 3 columns.
    """

    EMAfast = pd.Series(df[column].ewm( span = n_fast, min_periods = n_slow - 1).mean())
    EMAslow = pd.Series(df[column].ewm( span = n_slow, min_periods = n_slow - 1).mean())
    if column == 'y':
        MACD = pd.Series(EMAfast - EMAslow, name=f'MACD_{n_fast}_{n_slow}')
        MACDsign = pd.Series(
            MACD.ewm(span=9, min_periods=8).mean(),
            name=f'MACDsign_{n_fast}_{n_slow}',
        )
        MACDdiff = pd.Series(MACD - MACDsign, name=f'MACDdiff_{n_fast}_{n_slow}')
    else:
        MACD = pd.Series(EMAfast - EMAslow, name=f'{column}_MACD_{n_fast}_{n_slow}')
        MACDsign = pd.Series(
            MACD.ewm(span=9, min_periods=8).mean(),
            name=f'{column}_MACDsign_{n_fast}_{n_slow}',
        )
        MACDdiff = pd.Series(
            MACD - MACDsign, name=f'{column}_MACDdiff_{n_fast}_{n_slow}'
        )
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df

def BBANDS(df, n: int, column: str = 'y') -> pd.DataFrame:
    '''Adds two Bolllinger Band columns

    A Bollinger Band is a technical analysis tool defined by a
    set of trendlines plotted two standard deviations (positively and negatively)
    away from a simple moving average (SMA) of a security's price, but which can
    be adjusted to user preferences.

    Args:
        df: a pandas dataframe
        n: an integer on how many steps looking back.

        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added 2 BollingerBand columns.
    '''
    close = df[column]
    MA = pd.Series(close.rolling(n).mean())
    MSD = pd.Series(close.rolling(n).std())
    b1 = 4 * MSD / MA
    B1 = pd.Series(b1, name=f'BollingerBand1_{n}')
    df = df.join(B1)
    b2 = (close - MA + 2 * MSD) / (4 * MSD)
    B2 = pd.Series(b2, name=f'BollingerBand2_{n}')
    df = df.join(B2)
    return df

def TRIX(df, n: int, column: str = 'y') -> pd.DataFrame:
    '''Adds the TRIX indicator column

    The triple exponential average (TRIX) indicator is an oscillator
    used to identify oversold and overbought markets, and it can
    also be used as a momentum indicator.

    Args:
        df: a pandas dataframe
        n: an integer on how many steps looking back.

        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added TRIX column.
    '''
    close = df[column]
    EX1 = close.ewm( span = n, min_periods = n - 1).mean()
    EX2 = EX1.ewm( span = n, min_periods = n - 1).mean()
    EX3 = EX2.ewm( span = n, min_periods = n - 1).mean()

    i = 0
    ROC_l = [0]
    while i + 1 <= df.index[-1]:
        ROC = (EX3[i + 1] - EX3[i]) / EX3[i]
        ROC_l.append(ROC)
        i += 1
    Trix = pd.Series(ROC_l, name=f'TRIX_{n}')
    df = df.join(Trix)
    return df

def EMA(df, n: int, column: str = 'y') -> pd.DataFrame:
    '''Adds the Exponetial Moving Average column

    The exponential moving average (EMA) is a technical chart indicator that
    tracks the price of an investment (like a stock or commodity) over time.
    The EMA is a type of weighted moving average (WMA) that gives more
     weighting or importance to recent price data

    Args:
        df: a pandas dataframe
        n: an integer on how many steps looking back.

        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added EMA column.
    '''
    close = df[column]
    EMA = pd.Series(
        close.ewm(span=n, min_periods=n - 1).mean(), name=f'EMA_{n}'
    )
    df = df.join(EMA)
    return df

def TSI(df, r: int, s: int, column: str = 'y') -> pd.DataFrame:
    '''Adds the TSI column

    The true strength index (TSI) is a technical momentum oscillator
    used to identify trends and reversals

    Args:
        df: a pandas dataframe
        r: an integer on how many steps looking back, for window 1.
        s: an integer on how many steps looking back, for window 2.

        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added TSI column.
    '''
    close = df[column]
    M = pd.Series(close.diff(1))
    aM = abs(M)
    EMA1 = pd.Series(M.ewm( span = r, min_periods = r - 1).mean())
    aEMA1 = pd.Series(aM.ewm(span = r, min_periods = r - 1).mean())
    EMA2 = pd.Series(EMA1.ewm( span = s, min_periods = s - 1).mean())
    aEMA2 = pd.Series(aEMA1.ewm( span = s, min_periods = s - 1).mean())
    TSI = pd.Series(EMA2 / aEMA2, name=f'TSI_{r}_{s}')
    df = df.join(TSI)
    return df

def RSI(df, n: int, column: str = 'y') -> pd.DataFrame:
    '''
    Relative Strength Index (RSI)
    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    Args:
        df: a pandas dataframe
        n: an integer on how many steps looking back.

        column: Optional. If column is provided, will calculate based on provided column
            otherwise the column named y will be the target.

    Returns:
        A dataframe with all the columns from input df, and the added RSI column.

    '''
    close = df[column]
    diff = close.diff(1)
    up_direction = diff.where(diff > 0, 0.0)
    down_direction = -diff.where(diff < 0, 0.0)
    min_periods = n
    emaup = up_direction.ewm(alpha=1 / n, min_periods=min_periods, adjust=False).mean()
    emadn = down_direction.ewm(alpha=1 / n, min_periods=min_periods, adjust=False).mean()
    relative_strength = emaup / emadn
    rsi_col = pd.Series(
        np.where(emadn == 0, 100, 100 - (100 / (1 + relative_strength))),
        name=f'RSI_{n}',
    )
    df = df.join(rsi_col)
    return df
