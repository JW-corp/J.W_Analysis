from dataclasses import dataclass, field
from datetime import date
from typing import List

@dataclass
class User:
	id:int
	name:str
	birthdate:date
	admin:bool = False
	friends: List[int] = field(default_factory=list)
# Auto gen __init__, __repr__, __eq__


user1 = User(1, "user1", date(1990, 1, 1))
print(user1.friends)

user1.friends.append(2)
print(user1.friends)
