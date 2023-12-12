from typing import Annotated, List

from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .model import model_job


origins = [
    '*'
]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "DELETE", "PATCH", "PUT"],
    allow_headers=["Content-Type", "Set-Cookie",
                   "Access-Control-Allow-Headers",
                   "Access-Control-Allow-Origin",
                   "Authorization"],
)



@app.post('/machine-matching/')
def machine_matching(
    name_dealer_product: Annotated[str, Body(embed=True)]
) -> List[int]:
    result = model_job(name_dealer_product)
    return result[0]
