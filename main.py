   
import yfinance as yf
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from sqlmodel import create_engine, Session, SQLModel,Field
from utils.decorators import timing


   
from typing import List,Optional

@timing
def postprocess(data, tickers: List[str]) -> dict:
    """Post process results because of the way it has been organized."""

    data.dropna(axis=1, inplace=True)
    result = dict({ticker: dict() for ticker in tickers})

    try:
        for key, value in data.to_dict().items():
            if isinstance(key, tuple):
                (action, ticker) = key
                result[ticker][action] = value
            else:
                result[tickers[0]][key] = value
    except ValueError as e:
        raise e

    return result

class RequestDataInfo(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    tickers: str
    start: str
    end: str

app = FastAPI()

engine = create_engine("sqlite:///database.db")
SQLModel.metadata.create_all(engine)

@timing
@app.get("/get_prices")
def get_prices(tickers: str, init_date: str, init_end: str):
    yahoo_info = yf.download(" ".join(tickers.split(" ")), start=init_date, end=init_end)

    pos_data = postprocess(yahoo_info, tickers.split(" "))

    data = RequestDataInfo(tickers=tickers, start=init_date, end=init_date)

    with Session(engine) as session:
        session.add(data)
        session.commit()

    return jsonable_encoder(pos_data)