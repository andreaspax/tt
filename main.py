#this is a deployment test file
import fastapi

app = fastapi.FastAPI()

@app.get('/')

async def index(option: int):

    options = {1: "Hi there",
            2: "Hola amigo",
            3: "Salut amie",
            4: "Ciao tutti",}
    print("You selected '" + str(option) + "'")
    
    greeting_message = options[option]
    print(f"Option '{option}' was selected: '{greeting_message}'")
    return {"greeting":greeting_message}


if __name__ == "__main__":
    # Word you want to find the closest embedding for
    option = 2

   
    return_message = index(option) 

    # Print the result
    print(f"Option '{option}' was selected: '{return_message}'")
