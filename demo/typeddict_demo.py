from typing import TypedDict, List, Annotated

# Define a TypedDict for a Person with wrapped indices
class Person(TypedDict):
    name: str
    age: int
    hobbies: List[str]
    best_hobby_index: int
    best_hobby: Annotated[str, "hobbies[best_hobby_index]"]

# Create instances of the Person TypedDict
alice: Person = {
    "name": "Alice",
    "age": 30,
    "hobbies": ["reading", "hiking", "photography"],
    "best_hobby_index": 0,
    "best_hobby": "reading"
}

bob: Person = {
    "name": "Bob",
    "age": 25,
    "hobbies": ["gaming", "cooking"],
    "best_hobby_index": 0,
    "best_hobby": "gaming"
}

# Function that uses the Person TypedDict
def print_person_info(person: Person) -> None:
    print(f"Name: {person['name']}")
    print(f"Age: {person['age']}")
    print(f"Hobbies: {', '.join(person['hobbies'])}")
    print(f"Best hobby index: {person['best_hobby_index']}")
    print(f"Best hobby: {person['best_hobby']}")

# Function to update the best hobby based on the best_hobby_index
def update_best_hobby(person: Person) -> None:
    if 0 <= person['best_hobby_index'] < len(person['hobbies']):
        person['best_hobby'] = person['hobbies'][person['best_hobby_index']]
    else:
        person['best_hobby'] = "Invalid index"
        person['best_hobby_index'] = -1

# Demonstrate usage
if __name__ == "__main__":
    print("Alice's information:")
    print_person_info(alice)
    
    print("\nBob's information:")
    print_person_info(bob)

    # Update Bob's hobbies and best hobby index
    bob['hobbies'] = ["painting", "singing", "dancing"]
    bob['best_hobby_index'] = 1
    # update_best_hobby(bob)
    
    print("\nBob's updated information:")
    print_person_info(bob)

    bob['best_hobby'] = "reading"
    print_person_info(bob)

    # Demonstrate changing the best hobby index
    print("\nChanging Alice's best hobby index:")
    alice['best_hobby_index'] = 2
    # update_best_hobby(alice)
    print_person_info(alice)

    alice['best_hobby'] = "painting"
    print_person_info(alice)

    # Demonstrate error handling for invalid index
    print("\nTrying an invalid best hobby index for Alice:")
    alice['best_hobby_index'] = 5
    # update_best_hobby(alice)
    print_person_info(alice)

    # TypedDict helps catch errors at type-checking time
    # Uncomment the following line to see a type error:
    # invalid_person: Person = {"name": "Invalid", "age": "Not an int", "hobbies": [], "best_hobby_index": "0", "best_hobby": 123}  # This would raise a type error
