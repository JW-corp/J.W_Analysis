from dataclasses import dataclass
from datetime import date

@dataclass
class User:
    id:int
    name:str
    birthdate:date
    admin:bool = False

# Auto gen __init__, __repr__, __eq__


user1 = User(1, "user1", date(1990, 1, 1))
user2 = User(1, "user1", date(1990, 1, 1))
print(user1==user2)
print(user1)
