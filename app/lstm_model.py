import torch
from torch import nn


SEQUENCE_FEATURES = [
    "past_volatility_20d",
    "past_volatility_60d",
    "return_5d",
    "return_20d",
    "return_60d",
    "price_to_sma20",
    "price_to_sma50",
    "drawdown_60d",
    "market_volatility_20d",
]

SEQUENCE_LENGTH = 120
HIDDEN_SIZE = 32


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=len(SEQUENCE_FEATURES), hidden_size=HIDDEN_SIZE):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        prediction = self.fc(hidden[-1])
        return prediction.squeeze(1)
