import requests

endpoint = "http://127.0.0.1:8081/inference"

response = requests.post(url = endpoint,
              json = {"text" : "Publieke verantwoording begrippen vormen en beoordelingskaders in Bovens, M en T. Schillemans, Handboek publieke verantwoording. Den haag."}
              )


print(response.json())