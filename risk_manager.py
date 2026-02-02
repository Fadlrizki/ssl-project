"""
Risk Management Module for Trading Screener
Menghitung position sizing, stop loss, risk-reward ratio
"""

import pandas as pd
import numpy as np

class RiskManager:
    """
    Class untuk manajemen risiko trading
    """
    
    @staticmethod
    def calculate_position_size(capital, risk_per_trade, stop_loss_pct):
        """
        Calculate position size based on risk management principles
        
        Parameters:
        -----------
        capital : float
            Total trading capital
        risk_per_trade : float
            Percentage of capital to risk per trade (e.g., 2 for 2%)
        stop_loss_pct : float
            Stop loss percentage from entry (e.g., 5 for 5%)
        
        Returns:
        --------
        float : Position size in currency units
        """
        risk_amount = capital * (risk_per_trade / 100)
        if stop_loss_pct <= 0:
            raise ValueError("Stop loss percentage must be positive")
        position_size = risk_amount / (stop_loss_pct / 100)
        return position_size
    
    @staticmethod
    def calculate_stop_loss_price(entry_price, stop_loss_pct, direction="long"):
        """
        Calculate stop loss price
        
        Parameters:
        -----------
        entry_price : float
            Entry price
        stop_loss_pct : float
            Stop loss percentage
        direction : str
            "long" or "short"
        
        Returns:
        --------
        float : Stop loss price
        """
        if direction == "long":
            return entry_price * (1 - stop_loss_pct / 100)
        elif direction == "short":
            return entry_price * (1 + stop_loss_pct / 100)
        else:
            raise ValueError("Direction must be 'long' or 'short'")
    
    @staticmethod
    def calculate_target_price(entry_price, stop_loss_pct, risk_reward_ratio, direction="long"):
        """
        Calculate target price based on risk:reward ratio
        
        Parameters:
        -----------
        entry_price : float
            Entry price
        stop_loss_pct : float
            Stop loss percentage
        risk_reward_ratio : float
            Desired risk:reward ratio (e.g., 2 for 2:1)
        direction : str
            "long" or "short"
        
        Returns:
        --------
        float : Target price
        """
        if direction == "long":
            return entry_price * (1 + (stop_loss_pct * risk_reward_ratio / 100))
        elif direction == "short":
            return entry_price * (1 - (stop_loss_pct * risk_reward_ratio / 100))
        else:
            raise ValueError("Direction must be 'long' or 'short'")
    
    @staticmethod
    def calculate_risk_reward_ratio(entry_price, stop_loss_price, target_price, direction="long"):
        """
        Calculate risk:reward ratio
        
        Parameters:
        -----------
        entry_price : float
            Entry price
        stop_loss_price : float
            Stop loss price
        target_price : float
            Target price
        direction : str
        
        Returns:
        --------
        float : Risk:Reward ratio
        """
        if direction == "long":
            risk = entry_price - stop_loss_price
            reward = target_price - entry_price
        else:  # short
            risk = stop_loss_price - entry_price
            reward = entry_price - target_price
        
        if risk <= 0:
            return 0
        return reward / risk
    
    @staticmethod
    def calculate_max_drawdown(equity_curve):
        """
        Calculate maximum drawdown from equity curve
        
        Parameters:
        -----------
        equity_curve : list or pd.Series
            Series of equity values
        
        Returns:
        --------
        float : Maximum drawdown percentage
        """
        equity = pd.Series(equity_curve)
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max * 100
        return drawdown.min()
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.05, period='daily'):
        """
        Calculate Sharpe ratio
        
        Parameters:
        -----------
        returns : pd.Series
            Series of returns
        risk_free_rate : float
            Annual risk-free rate (default 5%)
        period : str
            'daily', 'weekly', 'monthly'
        
        Returns:
        --------
        float : Sharpe ratio
        """
        if returns.empty:
            return 0
        
        # Annualization factor
        if period == 'daily':
            factor = 252
        elif period == 'weekly':
            factor = 52
        elif period == 'monthly':
            factor = 12
        else:
            factor = 1
        
        excess_returns = returns - (risk_free_rate / factor)
        
        if returns.std() == 0:
            return 0
        
        sharpe = (excess_returns.mean() / returns.std()) * np.sqrt(factor)
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns, risk_free_rate=0.05, period='daily'):
        """
        Calculate Sortino ratio (downside risk only)
        
        Parameters:
        -----------
        returns : pd.Series
            Series of returns
        risk_free_rate : float
            Annual risk-free rate
        period : str
        
        Returns:
        --------
        float : Sortino ratio
        """
        if returns.empty:
            return 0
        
        # Annualization factor
        if period == 'daily':
            factor = 252
        elif period == 'weekly':
            factor = 52
        elif period == 'monthly':
            factor = 12
        else:
            factor = 1
        
        excess_returns = returns - (risk_free_rate / factor)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_dev = downside_returns.std()
        
        if downside_dev == 0:
            return 0
        
        sortino = (excess_returns.mean() / downside_dev) * np.sqrt(factor)
        return sortino
    
    @staticmethod
    def calculate_value_at_risk(returns, confidence_level=0.95):
        """
        Calculate Value at Risk (VaR)
        
        Parameters:
        -----------
        returns : pd.Series
            Series of returns
        confidence_level : float
            Confidence level (e.g., 0.95 for 95%)
        
        Returns:
        --------
        float : VaR (negative value represents loss)
        """
        if returns.empty:
            return 0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_expected_shortfall(returns, confidence_level=0.95):
        """
        Calculate Expected Shortfall (CVaR)
        
        Parameters:
        -----------
        returns : pd.Series
            Series of returns
        confidence_level : float
        
        Returns:
        --------
        float : Expected Shortfall
        """
        if returns.empty:
            return 0
        
        var = RiskManager.calculate_value_at_risk(returns, confidence_level)
        es = returns[returns <= var].mean()
        return es


class PositionSizer:
    """
    Advanced position sizing strategies
    """
    
    @staticmethod
    def fixed_fractional(capital, risk_per_trade, stop_loss_pct):
        """Fixed fractional position sizing"""
        return RiskManager.calculate_position_size(capital, risk_per_trade, stop_loss_pct)
    
    @staticmethod
    def kelly_criterion(win_rate, avg_win_pct, avg_loss_pct):
        """
        Kelly Criterion position sizing
        
        Parameters:
        -----------
        win_rate : float
            Win rate (0-1)
        avg_win_pct : float
            Average win percentage
        avg_loss_pct : float
            Average loss percentage
        
        Returns:
        --------
        float : Kelly percentage
        """
        if avg_loss_pct == 0:
            return 0
        
        b = avg_win_pct / avg_loss_pct
        kelly = win_rate - ((1 - win_rate) / b)
        return kelly * 100  # Convert to percentage
    
    @staticmethod
    def optimal_f(capital, win_rate, avg_win, avg_loss):
        """
        Optimal F position sizing (Ralph Vince)
        
        Parameters:
        -----------
        capital : float
            Trading capital
        win_rate : float
            Win rate (0-1)
        avg_win : float
            Average win amount
        avg_loss : float
            Average loss amount
        
        Returns:
        --------
        float : Optimal F position size
        """
        # Simplified version
        if avg_loss == 0:
            return 0
        
        # Calculate optimal fraction
        optimal_f = win_rate - ((1 - win_rate) / (avg_win / abs(avg_loss)))
        
        # Limit to reasonable range
        optimal_f = max(0.01, min(optimal_f, 0.25))
        
        return capital * optimal_f