#! /bin/bash

virtualenv -q -p /usr/bin/python3.5 $1
source $1/bin/activate
$1/bin/pip install -r requirements.txt

python3 -m spacy download en

pip install speechRecognition


