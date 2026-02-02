"""
Performance Tracking Module for Trading Screener
Melacak kinerja trading, menyimpan history, dan menghasilkan laporan
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional
from risk_manager import RiskManager


class TradeRecord:
    """Class untuk merekam detail setiap trade"""
    
    def __init__(self, symbol, entry_price, exit_price, entry_time, exit_time, 
                 quantity, direction="long", stop_loss=None, target=None,
                 commission=0.0015, slippage=0.001):
        
        self.symbol = symbol
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.quantity = quantity
        self.direction = direction
        self.stop_loss = stop_loss
        self.target = target
        self.commission = commission  # 0.15% untuk IDX
        self.slippage = slippage  # 0.1% slippage
        
        # Calculate metrics
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate all trade metrics"""
        # Gross P&L
        if self.direction == "long":
            self.gross_pnl = (self.exit_price - self.entry_price) * self.quantity
            self.gross_pnl_pct = (self.exit_price / self.entry_price - 1) * 100
        else:  # short
            self.gross_pnl = (self.entry_price - self.exit_price) * self.quantity
            self.gross_pnl_pct = (1 - self.exit_price / self.entry_price) * 100
        
        # Commission & slippage costs
        entry_cost = self.entry_price * self.quantity * self.commission
        exit_cost = self.exit_price * self.quantity * self.commission
        slippage_cost = (self.entry_price + self.exit_price) * self.quantity * self.slippage / 2
        
        self.commission_cost = entry_cost + exit_cost
        self.slippage_cost = slippage_cost
        self.total_cost = self.commission_cost + self.slippage_cost
        
        # Net P&L
        self.net_pnl = self.gross_pnl - self.total_cost
        self.net_pnl_pct = (self.net_pnl / (self.entry_price * self.quantity)) * 100
        
        # Duration
        self.duration = (self.exit_time - self.entry_time).days
        
        # Win/loss
        self.win = self.net_pnl > 0
        
        # Risk metrics if stop loss and target are provided
        if self.stop_loss and self.target:
            self.risk_reward_ratio = RiskManager.calculate_risk_reward_ratio(
                self.entry_price, self.stop_loss, self.target, self.direction
            )
        else:
            self.risk_reward_ratio = None
    
    def to_dict(self):
        """Convert to dictionary for storage"""
        return {
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'quantity': self.quantity,
            'direction': self.direction,
            'stop_loss': self.stop_loss,
            'target': self.target,
            'gross_pnl': self.gross_pnl,
            'gross_pnl_pct': self.gross_pnl_pct,
            'net_pnl': self.net_pnl,
            'net_pnl_pct': self.net_pnl_pct,
            'commission_cost': self.commission_cost,
            'slippage_cost': self.slippage_cost,
            'total_cost': self.total_cost,
            'duration': self.duration,
            'win': self.win,
            'risk_reward_ratio': self.risk_reward_ratio
        }


class PerformanceTracker:
    """
    Main class untuk tracking performance trading
    """
    
    def __init__(self, initial_capital=10000000, save_path="performance_data"):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades: List[TradeRecord] = []
        self.equity_curve = [initial_capital]
        self.daily_returns = []
        self.save_path = save_path
        
        # Create save directory if not exists
        os.makedirs(save_path, exist_ok=True)
        
        # Load existing data if available
        self.load_data()
    
    def add_trade(self, symbol, entry_price, exit_price, 
                  entry_time, exit_time, quantity, direction="long",
                  stop_loss=None, target=None):
        """
        Add a completed trade to tracker
        
        Returns:
        --------
        TradeRecord : Trade record object
        """
        trade = TradeRecord(
            symbol=symbol,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            quantity=quantity,
            direction=direction,
            stop_loss=stop_loss,
            target=target
        )
        
        # Update capital
        self.current_capital += trade.net_pnl
        self.equity_curve.append(self.current_capital)
        
        # Calculate daily return if enough time has passed
        if len(self.equity_curve) >= 2:
            daily_return = (self.equity_curve[-1] / self.equity_curve[-2] - 1)
            self.daily_returns.append(daily_return)
        
        self.trades.append(trade)
        
        # Auto-save after adding trade
        self.save_data()
        
        return trade
    
    def add_trade_from_dict(self, trade_dict):
        """Add trade from dictionary"""
        trade = TradeRecord(**trade_dict)
        self.trades.append(trade)
        self.current_capital += trade.net_pnl
        self.equity_curve.append(self.current_capital)
        self.save_data()
        return trade
    
    def get_summary(self):
        """Get comprehensive performance summary"""
        if not self.trades:
            return {}
        
        trades_df = self.get_trades_df()
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['win']])
        losing_trades = len(trades_df[~trades_df['win']])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_gross_pnl = trades_df['gross_pnl'].sum()
        total_net_pnl = trades_df['net_pnl'].sum()
        total_commission = trades_df['commission_cost'].sum()
        total_slippage = trades_df['slippage_cost'].sum()
        total_cost = trades_df['total_cost'].sum()
        
        # Percentage returns
        total_return_pct = (self.current_capital / self.initial_capital - 1) * 100
        annual_return_pct = self._calculate_annual_return()
        
        # Win/Loss metrics
        avg_win = trades_df[trades_df['win']]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[~trades_df['win']]['net_pnl'].mean() if losing_trades > 0 else 0
        avg_win_pct = trades_df[trades_df['win']]['net_pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss_pct = trades_df[~trades_df['win']]['net_pnl_pct'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        largest_win = trades_df['net_pnl'].max()
        largest_loss = trades_df['net_pnl'].min()
        profit_factor = abs(trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum() / 
                           trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum()) if losing_trades > 0 else float('inf')
        
        # Advanced metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        sortino_ratio = self._calculate_sortino_ratio()
        max_drawdown = RiskManager.calculate_max_drawdown(self.equity_curve)
        recovery_factor = abs(total_net_pnl / max_drawdown) if max_drawdown < 0 else float('inf')
        
        # Win streaks
        win_streak, loss_streak = self._calculate_streaks()
        
        summary = {
            # Capital metrics
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'total_return_idr': total_net_pnl,
            'total_return_pct': total_return_pct,
            'annual_return_pct': annual_return_pct,
            
            # Trade statistics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_trades_per_day': self._calculate_avg_trades_per_day(),
            'avg_trade_duration': trades_df['duration'].mean(),
            
            # P&L metrics
            'total_gross_pnl': total_gross_pnl,
            'total_net_pnl': total_net_pnl,
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'total_cost': total_cost,
            'cost_percentage': (total_cost / abs(total_gross_pnl)) * 100 if total_gross_pnl != 0 else 0,
            
            # Win/Loss metrics
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'profit_factor': profit_factor,
            'expectancy': (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss),
            
            # Risk metrics
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'recovery_factor': recovery_factor,
            'var_95': RiskManager.calculate_value_at_risk(pd.Series(self.daily_returns), 0.95),
            'cvar_95': RiskManager.calculate_expected_shortfall(pd.Series(self.daily_returns), 0.95),
            
            # Streaks
            'longest_win_streak': win_streak,
            'longest_loss_streak': loss_streak,
            
            # Performance ratios
            'calmar_ratio': annual_return_pct / abs(max_drawdown) if max_drawdown < 0 else float('inf'),
            'mar_ratio': total_return_pct / abs(max_drawdown) if max_drawdown < 0 else float('inf'),
        }
        
        return summary
    
    def _calculate_annual_return(self):
        """Calculate annualized return"""
        if not self.trades:
            return 0
        
        first_trade_date = min(trade.entry_time for trade in self.trades)
        last_trade_date = max(trade.exit_time for trade in self.trades)
        
        days_active = (last_trade_date - first_trade_date).days
        if days_active <= 0:
            days_active = 1
        
        total_return = (self.current_capital / self.initial_capital - 1)
        annual_return = ((1 + total_return) ** (365 / days_active) - 1) * 100
        
        return annual_return
    
    def _calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio"""
        if len(self.daily_returns) < 2:
            return 0
        
        returns_series = pd.Series(self.daily_returns)
        return RiskManager.calculate_sharpe_ratio(returns_series)
    
    def _calculate_sortino_ratio(self):
        """Calculate Sortino ratio"""
        if len(self.daily_returns) < 2:
            return 0
        
        returns_series = pd.Series(self.daily_returns)
        return RiskManager.calculate_sortino_ratio(returns_series)
    
    def _calculate_avg_trades_per_day(self):
        """Calculate average trades per day"""
        if not self.trades:
            return 0
        
        dates = [trade.entry_time.date() for trade in self.trades]
        unique_days = len(set(dates))
        
        return len(self.trades) / unique_days if unique_days > 0 else 0
    
    def _calculate_streaks(self):
        """Calculate longest win and loss streaks"""
        if not self.trades:
            return 0, 0
        
        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        
        for trade in self.trades:
            if trade.win:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return max_win_streak, max_loss_streak
    
    def get_trades_df(self):
        """Get all trades as DataFrame"""
        if not self.trades:
            return pd.DataFrame()
        
        trades_list = [trade.to_dict() for trade in self.trades]
        df = pd.DataFrame(trades_list)
        
        # Format datetime columns
        if not df.empty:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['exit_time'] = pd.to_datetime(df['exit_time'])
        
        return df
    
    def get_monthly_performance(self):
        """Get performance grouped by month"""
        trades_df = self.get_trades_df()
        
        if trades_df.empty:
            return pd.DataFrame()
        
        trades_df['year_month'] = trades_df['entry_time'].dt.strftime('%Y-%m')
        
        monthly = trades_df.groupby('year_month').agg({
            'net_pnl': 'sum',
            'net_pnl_pct': 'mean',
            'symbol': 'count'
        }).rename(columns={
            'net_pnl': 'total_pnl',
            'net_pnl_pct': 'avg_pnl_pct',
            'symbol': 'trades'
        })
        
        monthly['win_rate'] = trades_df.groupby('year_month')['win'].mean() * 100
        monthly['cumulative_pnl'] = monthly['total_pnl'].cumsum()
        
        return monthly
    
    def get_symbol_performance(self):
        """Get performance grouped by symbol"""
        trades_df = self.get_trades_df()
        
        if trades_df.empty:
            return pd.DataFrame()
        
        symbol_perf = trades_df.groupby('symbol').agg({
            'net_pnl': ['sum', 'mean', 'count'],
            'net_pnl_pct': 'mean',
            'win': 'mean'
        })
        
        symbol_perf.columns = ['total_pnl', 'avg_pnl', 'trades', 'avg_pnl_pct', 'win_rate']
        symbol_perf['win_rate'] = symbol_perf['win_rate'] * 100
        symbol_perf = symbol_perf.sort_values('total_pnl', ascending=False)
        
        return symbol_perf
    
    def save_data(self, filename="performance_data.json"):
        """Save performance data to file"""
        save_file = os.path.join(self.save_path, filename)
        
        data = {
            'initial_capital': self.initial_capital,
            'current_capital': self.current_capital,
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns,
            'trades': [trade.to_dict() for trade in self.trades]
        }
        
        with open(save_file, 'w') as f:
            json.dump(data, f, default=str, indent=2)
    
    def load_data(self, filename="performance_data.json"):
        """Load performance data from file"""
        load_file = os.path.join(self.save_path, filename)
        
        if os.path.exists(load_file):
            try:
                with open(load_file, 'r') as f:
                    data = json.load(f)
                
                self.initial_capital = data.get('initial_capital', self.initial_capital)
                self.current_capital = data.get('current_capital', self.current_capital)
                self.equity_curve = data.get('equity_curve', self.equity_curve)
                self.daily_returns = data.get('daily_returns', self.daily_returns)
                
                # Load trades
                trades_data = data.get('trades', [])
                for trade_dict in trades_data:
                    # Convert string dates back to datetime
                    trade_dict['entry_time'] = pd.to_datetime(trade_dict['entry_time'])
                    trade_dict['exit_time'] = pd.to_datetime(trade_dict['exit_time'])
                    self.trades.append(TradeRecord(**trade_dict))
                
                print(f"Loaded {len(self.trades)} trades from {load_file}")
                
            except Exception as e:
                print(f"Error loading performance data: {e}")
    
    def reset(self):
        """Reset all performance data"""
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self.current_capital = self.initial_capital
        self.daily_returns = []
        print("Performance tracker reset")
    
    def export_report(self, format='csv'):
        """Export performance report"""
        if format == 'csv':
            # Export trades
            trades_df = self.get_trades_df()
            if not trades_df.empty:
                trades_df.to_csv(os.path.join(self.save_path, 'trades_report.csv'), index=False)
            
            # Export summary
            summary = self.get_summary()
            summary_df = pd.DataFrame([summary])
            summary_df.to_csv(os.path.join(self.save_path, 'summary_report.csv'), index=False)
            
            print(f"Reports exported to {self.save_path}")
        
        elif format == 'excel':
            with pd.ExcelWriter(os.path.join(self.save_path, 'performance_report.xlsx')) as writer:
                # Trades sheet
                trades_df = self.get_trades_df()
                if not trades_df.empty:
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Summary sheet
                summary = self.get_summary()
                summary_df = pd.DataFrame([summary])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Monthly performance
                monthly_df = self.get_monthly_performance()
                if not monthly_df.empty:
                    monthly_df.to_excel(writer, sheet_name='Monthly')
                
                # Symbol performance
                symbol_df = self.get_symbol_performance()
                if not symbol_df.empty:
                    symbol_df.to_excel(writer, sheet_name='By Symbol')
            
            print(f"Excel report exported to {self.save_path}")