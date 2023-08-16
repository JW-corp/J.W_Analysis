from datetime import date

class User:
	def __init__(
			self,id:int, name: str, birthdate:date, admin:bool = False
			) -> None:

		self.id = id
		self.name = name
		self.birthdate = birthdate
		self.admin = admin

	# Show the class name and the attributes
	def __repr__(self):
		return (
			self.__class__.__qualname__ + f"(id={self.id!r}, name={self.name!r}, "
			f"birthdate={self.birthdate!r}, admin={self.admin!r})"
		)

	def __eq__(self,other):
		if other.__class__ is self.__class__:
			return (self.id, self.name, self.birthdate, self.admin) == (other.id, other.name, other.birthdate, other.admin)
		return NotImplemented

user1 = User(id=1, name="John Doe", birthdate=date(1990, 1, 1))
user2 = User(id=1, name="John Doe", birthdate=date(1990, 1, 1))

print(user1==user2)
