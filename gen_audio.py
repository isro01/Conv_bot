#!pip install gtts
#!apt install mpg123
from gtts import gTTS

import os

def string_to_audio(input_string, delete):
  language = 'en'
  gen_audio = gTTS(text = input_string, lang=language, slow=False)
  gen_audio.save("Output.mp3")
  os.system("mpg123 output.mp3")
  if (delete == True):
    os.remove("Output.mp3")

# string_to_audio("hello", False)

