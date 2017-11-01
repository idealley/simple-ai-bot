# Small Bot API powered by its own Neural net.
This is a small experiment, but it works. Of course it could be improved, but that is just a simple Proof of Concept to classify sentenses.

## How to use
Lauch the Jupyter Notebook:

* `clone` the repo
* `cd` into the repo
```
>>> jupyter notebook
```
### or

* `clone` the repo
* `cd` into the repo
* run `python train.py`
* run `python api.py`

Send a `POST` request to `localhost:1234` do not forget the header `Content-Type application/json`
The API will return what it understood.

## Training
Un comment the lines that correspond to the different trainings in
* `/api.py`
* `/train.py` 
* `/brain.py`

I aggree this could be better... 

You can also play with the input parametres of the neural net the defaults are:
* `hidden_neurons=20`
* `alpha=0.1`
* `epochs=100000` 
* `dropout=False`
* `dropout_percent=0.2`

# Next Step
* Turn this into nicer code
* Integrate with GRAKN.AI
* Write a tutorial