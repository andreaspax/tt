#this is a deployment test file
import fastapi

app = fastapi.FastAPI()

@app.get('/')

def index (option):
    options = {1: "Hi there",
            2: "Hola amigo",
            3: "Salut amie",
            4: "Ciao tutti"}
    print("You selected '"+option+"'")

    return print(options[option])



