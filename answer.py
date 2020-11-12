"""
Case study

Step 1 - Prepare list of 25 largest components of Russell1000
- Download and parse Russell1000 Membership list
  on June 29 2020 from ftserussell.com
- Download and parse Russell1000 Quarterly Membership
  Weights on June 30 2020 from ftserussell.com
- Find top 25 instruments by weight
Step 2 - Download historical daily data from Yahoo Finance
- For 25 instruments
- For ^RUI
Step 3 - Benchmark
- Get returns matrix for 25 instruments
- Get returns series for ^RUI
- Build benchmark - market-weighted index using 25 instruments
- Find tracking error for ^RUI and benchmark on test period
- Save ^RUI vs Benchmark index price plot
Step 4 - My Index
- Get returns matrix and ^RUI returns train samples
- Find optimal weights for train sample by solving Non-negative least squares
  optimization problem
- Build My_index using optimized weights
- Find tracking error for ^RUI and My_index on test period
- Save weights
- Save  ^RUI vs My_index price plot

"""

from pathlib import Path

import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tabula
import yfinance as yf


WEIGHT_REPORT_LINK = "https://research.ftserussell.com/analytics/factsheets/Home/DownloadConstituentsWeights/?indexdetails=US1000"
MEMBERSHIP_REPORT_LINK = "https://content.ftserussell.com/sites/default/files/ru1000_membershiplist_20200629.pdf"
RUSSEL1000_TICKER = '^RUI'


def download_weights_report(report_link: str = WEIGHT_REPORT_LINK) -> pd.DataFrame():
    """
    Download Quarterly Membership Weights report

    """
    tables_list = tabula.read_pdf(
        report_link, pages="all", multiple_tables=True)
    weights_df = pd.concat(tables_list).rename(
        columns={'Russell 1000®': 'Company', 'Weight(%)': 'Weight'})
    weights_df.Company = weights_df.Company.apply(lambda x: x.lower())
    return weights_df


def download_membership_report(report_link: str = MEMBERSHIP_REPORT_LINK) -> pd.DataFrame():
    """
    Download Membership report

    """
    tables_list = tabula.read_pdf(
        report_link, pages="all", multiple_tables=True)
    membership_df = pd.concat(tables_list)
    membership_df1 = membership_df[['Company', 'Ticker']]
    membership_df2 = membership_df[['Ticker.1', 'Unnamed: 0']]
    membership_df2 = membership_df2.rename(columns={'Ticker.1': 'Company',
                                                    'Unnamed: 0': 'Ticker'})
    membership_df = pd.concat([membership_df1, membership_df2])
    membership_df = membership_df.dropna().reset_index(drop=True)
    membership_df.Company = membership_df.Company.apply(lambda x: x.lower())
    membership_df.Ticker = membership_df.Ticker.apply(
        lambda x: x.replace('.', '-'))
    return membership_df


def download_historical(ticker: str,
                        start_date: str,
                        end_date: str) -> pd.DataFrame:
    """
    Download historical daily data for ticker from Yahoo Finance

    """
    data_df = yf.Ticker(ticker).history(
        start=start_date, end=end_date, interval='1d', auto_adjust=False)
    data_df['Returns'] = data_df['Adj Close'].pct_change()
    return data_df


def create_weight_matrix(weights: np.array,
                         returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weight matrix for given returns matrix

    """
    weights_df = pd.DataFrame().reindex_like(returns_df)
    weights_df.iloc[0] = weights
    # Use same weights all test period
    weights_df = weights_df.ffill()
    return weights_df


def tracking_error(benchmark_returns: pd.Series,
                   portfolio_returns: pd.Series) -> float:
    """
    Find tracking error using portfolio returns and benchmark returns

    """
    return (benchmark_returns-portfolio_returns).std(ddof=1)


def get_portfolio_returns(weights_matrix: pd.DataFrame,
                          returns_matrix: pd.DataFrame) -> pd.Series:
    """
    Generate portfolio returns using instruments returns
    and weights matrix
    """
    return (weights_matrix*returns_matrix).sum(axis=1)


def returns_to_prices(returns: pd.Series, start_price: float) -> pd.Series:
    """
    Transform returns series to price series
    """
    return returns.add(1).cumprod().mul(start_price)


def save_compare_plot(series1: pd.Series,
                      series2: pd.Series,
                      label1: str,
                      label2: str,
                      title: str,
                      file_name: str) -> None:
    """
    Save two series plot
    """
    _, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(series1, label=label1)
    ax.plot(series2, label=label2)
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title(title)
    plt.savefig(f'{file_name}.png')


def nlls_weights_fit(A: np.ndarray,
                     y: np.ndarray,
                     lmbda: float = 0.0,
                     min_weight: float = 0.0) -> np.ndarray:
    """
    Solve Non-negative least-square problem

    minimize 1/2 x'*Q*x + c'*x

    where Q = A'*A + lambda*I , c = -A'*y
    subject to:
                sum x = 1
                x >= min_weight

    :param A: matrix n x m
    :param y: vector n x 1
    :param lmbda: regulzarization parameter
    :param min_weight: minimum x value constraint
    :return: answer vector x if exists, else zero vector

    """
    n, m = A.shape
    Q = A.T @ A + lmbda * np.eye(m)
    c = - A.T @ y
    x = cp.Variable(m)
    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x),
                      [cp.sum(x) == 1,
                       x >= min_weight])
    result = prob.solve()
    if np.isfinite(result):
        return x.value
    print("Can't solve optimization problem.")
    return np.zeros(m)


def main() -> None:
    """Main function"""

    # STEP 1 Download and save Russell 1000 reports

    print('Download and save Weights report...')
    weights_df = download_weights_report()
    weights_df.to_csv('weights_report.csv', index=False)
    print('Download and save Membership report...')
    membership_df = download_membership_report()
    membership_df.to_csv('membership_report.csv', index=False)
    # Merge reports to get "ticker - weight" table
    ticker_weight_df = pd.merge(
        weights_df, membership_df, how='left', on=['Company'])
    # Sort by weights and get top 25 companies by weight
    print('Get top 25 companies by weight and save...')
    ticker_weight_df = ticker_weight_df.sort_values(
        by='Weight', ascending=False).reset_index(drop=True)
    top25_df = ticker_weight_df.iloc[0:25]
    top25_df.to_csv('top25.csv', index=False)

    # STEP 2 Download and save historical data

    print('Download and save historical data...')
    start_date = '2018-01-01'
    end_date = '2020-08-01'
    Path('data').mkdir(parents=True, exist_ok=True)
    data_path = Path('data').resolve()
    tickers_list = top25_df.Ticker.to_list() + [RUSSEL1000_TICKER]
    for ticker in tickers_list:
        ticker_df = download_historical(ticker, start_date, end_date)
        ticker_df.to_csv(data_path.joinpath(f"{ticker}.csv"))

    # STEP 3 Build Benchmark

    train_period_start = '2020-01-01'
    train_period_end = '2020-06-30'
    test_period_start = '2020-07-01'

    # Get returns matrix for 25 instruments
    top25_df = pd.read_csv('top25.csv')
    returns_df = pd.DataFrame()
    for ticker in top25_df.Ticker.to_list():
        ticker_data_path = data_path.joinpath(f'{ticker}.csv')
        ticker_data_df = pd.read_csv(ticker_data_path)
        ticker_data_df.Date = pd.to_datetime(ticker_data_df.Date)
        ticker_data_df = ticker_data_df.set_index('Date')
        returns_df[ticker] = ticker_data_df['Returns']
    # Get returns for ^RUI
    ticker_data_path = data_path.joinpath(f'{RUSSEL1000_TICKER}.csv')
    rui_data_df = pd.read_csv(ticker_data_path)
    rui_data_df.Date = pd.to_datetime(rui_data_df.Date)
    rui_data_df = rui_data_df.set_index('Date')
    rui_returns_df = rui_data_df['Returns']
    rui_close_df = rui_data_df['Close']

    # Generate Benchmark index
    # Test sample
    rui_returns_df_test = rui_returns_df.loc[test_period_start:]
    returns_df_test = returns_df.loc[test_period_start:]
    # Use Russell1000 weights
    weights = top25_df[['Ticker', 'Weight']].set_index('Ticker').Weight
    # L1-normalize weights
    weights = weights / weights.abs().sum()
    # weights_matrix
    weights_df = create_weight_matrix(weights, returns_df_test)
    # Find benchmark returns
    benchmark_returns = get_portfolio_returns(weights_df, returns_df_test)
    # Find tracking error for top25 market-weighted index
    benchmark_te = tracking_error(rui_returns_df_test, benchmark_returns)
    print('='*20)
    print('Benchmark Index: top 25 Russell1000 components')
    print('Weights: L1-normalized Russell1000 weights')
    print(f'Tracking error: {benchmark_te}')

    # Create plots
    rui_close_train_end = rui_close_df.loc[train_period_end]
    rui_close_test_df = rui_close_df.loc[test_period_start:]
    # Get Benchmark price
    benchmark_price = returns_to_prices(benchmark_returns, rui_close_train_end)
    # Save plot
    save_compare_plot(rui_close_test_df, benchmark_price, '^RUI close price',
                      'Benchmark price', 'Benchmark vs Russell 1000 Index',
                      'benchmark_vs_russel')

    # STEP 4 Build My Index

    # Generate My_Index
    # Test sample
    returns_df_test = returns_df.loc[test_period_start:]
    rui_returns_df_test = rui_returns_df.loc[test_period_start:]
    # Train sample
    returns_df_train = returns_df.loc[train_period_start:train_period_end]
    rui_returns_df_train = rui_returns_df.loc[train_period_start:train_period_end]

    # Find optimal weights
    lmbda = 0.5
    min_weight = 0.01
    weights = nlls_weights_fit(returns_df_train.to_numpy(),
                               rui_returns_df_train.to_numpy(),
                               lmbda=lmbda,
                               min_weight=min_weight)

    # Weights_matrix
    weights_df = create_weight_matrix(weights, returns_df_test)
    # Find benchmark returns
    myindex_returns = get_portfolio_returns(weights_df, returns_df_test)
    # Find tracking error for top25 market-weighted index
    myindex_te = tracking_error(rui_returns_df_test, myindex_returns)
    print('='*20)
    print('My_Index: top 25 Russell1000 components')
    print('Weights: NLLS optimized weights')
    print(f'Train period: {train_period_start} - {train_period_end}')
    print(f'Tracking error: {myindex_te}')

    # Create plots
    myindex_price = returns_to_prices(myindex_returns, rui_close_train_end)
    save_compare_plot(rui_close_test_df, myindex_price, '^RUI close price',
                      'My_Index price', 'My_Index vs Russell 1000 Index',
                      'my_index_vs_russel')
    return


if __name__ == '__main__':
    main()
