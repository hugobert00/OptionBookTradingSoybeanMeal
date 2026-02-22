#=====================
#Savings management : 

#=====================
import pandas as pd 
import numpy as np 
from pathlib import Path

BASE_DATA_PATH = Path("/Users/hugoberthelier/Desktop/PricerLB_Soymeal/books") #<---- replace this by your desktop path
COLUMNS = [
    "trade_id",
    "date",
    "underlying",
    "type",
    "expiry",
    "lots",
    "quantity",
    "strike",
    "price/premium",
    "cost",
    "units",
]

def get_openbook_path(account_id: str) -> Path:
    return BASE_DATA_PATH / str(account_id) / "book.xlsx"



def load_open_positions(account_id: str) -> pd.DataFrame:
    account_id = str(account_id)
    path = get_openbook_path(account_id)

    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)

    df = pd.read_excel(path)


    return df




def save_open_positions(account_id: str, df: pd.DataFrame) -> None:
    path = get_openbook_path(account_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.reindex(columns=COLUMNS)

    try:
        df.to_excel(path, index=False)

    except Exception as e:
        raise RuntimeError(
            f"Unable to save open positions for account {account_id}"
        ) from e



def emergency_saving():
    return()


def clear_position():
    return 

