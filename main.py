#this is a deployment test file
import fastapi

app = fastapi.FastAPI()

@app.post('/option')

async def index(request: Request):
    body = await request.json()
    option = body["option"]

    options = {1: "Hi there",
            2: "Hola amigo",
            3: "Salut amie",
            4: "Ciao tutti",}
    print("You selected '" + str(option) + "'")
    
    greeting_message = options[option]
    return {"greeting":greeting_message}


if __name__ == "__main__":
    # Word you want to find the closest embedding for
    option = 2

   
    return_message = index(option) 

    # Print the result
    print(f"Option '{option}' was selected: '{return_message}'")
