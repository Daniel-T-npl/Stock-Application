import pandas as pd
import matplotlib.pyplot as plt
import os

# Robust path: always find data relative to this script
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.abspath(os.path.join(script_dir, '../data'))

strategy_files = [
    ('Trend Following', 'trend_following_all.csv'),
    ('Mean Reversion', 'mean_reversion_all.csv'),
    ('Breakout', 'breakout_all.csv'),
    ('Reversal', 'reversal_all.csv'),
    ('Divergence', 'divergence_all.csv'),
    ('Momentum Breakout', 'momentum_breakout_all.csv'),
    ('VWAP Reversion', 'vwap_reversion_all.csv'),
    ('Smart Money Divergence', 'smart_money_divergence_all.csv'),
    ('Volatility Expansion Fade', 'volatility_expansion_fade_all.csv'),
    ('Trend Pullback', 'trend_pullback_all.csv'),
    ('Cloud Crossover', 'cloud_crossover_all.csv'),
    ('Composite Score', 'composite_score_all.csv'),
]

all_summaries = {}

for strat_name, strat_file in strategy_files:
    strategy_path = os.path.join(data_dir, strat_file)
    if not os.path.exists(strategy_path):
        print(f"File not found: {strategy_path}")
        continue
    print(f"\n===== {strat_name} ({strat_file}) =====")
    df = pd.read_csv(strategy_path)
    profits = {}
    trades = {}
    current_profit = 0
    current_trades = 0
    buy_price = None
    for idx, row in df.iterrows():
        symbol = row['symbol']
        action = row['action']
        price = row['price']
        if not isinstance(symbol, str) or not isinstance(action, str):
            continue
        if pd.isna(symbol) or pd.isna(action):
            continue
        if action == 'Buy':
            buy_price = price
        elif action == 'Sell' and buy_price is not None:
            try:
                profit = float(price) - float(buy_price)
                current_profit += profit
                current_trades += 1
            except Exception:
                pass
            buy_price = None
        elif action == 'STOCK_PROFIT':
            profits[symbol] = current_profit
            trades[symbol] = current_trades
            current_profit = 0
            current_trades = 0
            buy_price = None
    summary_df = pd.DataFrame({
        'symbol': list(profits.keys()),
        'profit': list(profits.values()),
        'trades': [trades[s] for s in profits.keys()]
    })
    summary_df = summary_df.sort_values('profit', ascending=False)
    all_summaries[strat_name] = summary_df.set_index('symbol')
    print('Top 10 stocks by profit:')
    print(summary_df.head(10))
    print('\nBottom 10 stocks by profit:')
    print(summary_df.tail(10))
    avg_profit = summary_df['profit'].mean()
    avg_trades = summary_df['trades'].mean()
    print(f'\nAverage profit: {avg_profit}')
    print(f'Average number of trades: {avg_trades}')
    # Plot Profit Distribution
    plt.figure(figsize=(10,5))
    plt.hist(summary_df['profit'].dropna(), bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Distribution of Per-Stock Profits: {strat_name}')
    plt.xlabel('Total Profit')
    plt.ylabel('Number of Stocks')
    plt.tight_layout()
    plt.show()
    # Number of Trades per Stock (Top 20)
    trade_counts = summary_df['trades'].sort_values(ascending=False)
    print('\nTop 10 stocks by number of trades:')
    print(trade_counts.head(10))
    plt.figure(figsize=(12,6))
    trade_counts.head(20).plot(kind='bar')
    plt.title(f'Number of Trades per Stock (Top 20): {strat_name}')
    plt.xlabel('Stock Symbol')
    plt.ylabel('Number of Trades')
    plt.tight_layout()
    plt.show()

# --- Comparison Table ---
# Merge all summaries on symbol to compare per-stock profit across strategies
comparison = None
for strat_name, summary in all_summaries.items():
    colname = strat_name.replace(' ', '_').lower() + '_profit'
    s = summary['profit'].rename(colname)
    if comparison is None:
        comparison = s.to_frame()
    else:
        comparison = comparison.join(s, how='outer')

if comparison is not None:
    print("\n===== Per-Stock Profit Comparison Table =====")
    print(comparison.head(20))
    # Show top/bottom stocks for each strategy
    for col in comparison.columns:
        print(f"\nTop 5 stocks for {col}:")
        print(comparison[col].sort_values(ascending=False).head(5))
        print(f"\nBottom 5 stocks for {col}:")
        print(comparison[col].sort_values().head(5))
    # Boxplot for profit distributions
    plt.figure(figsize=(10,6))
    comparison.boxplot()
    plt.title('Per-Stock Profit Distribution Across Strategies')
    plt.ylabel('Profit')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show() 