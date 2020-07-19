import aiml
import os

kernel = aiml.Kernel()


if os.path.isfile("bot_brain.brn"):
    kernel.bootstrap(brainFile = "bot_brain.brn")
else:
    kernel.bootstrap(learnFiles = "Brain.xml", commands = "LOAD BASIC CHAT")
    kernel.saveBrain("bot_brain.brn")

# kernel now ready for use

while True:
    respond = input("USER > ")

    if respond.lower() == "bye" :
        print("Would you like to give us a feedback.")

        respond = input("> ")
        if respond.lower() == "yes" :
    
            respond = input("Feedback : ")

            print("Thanks, for your feedback sir !")

            break
        
        if respond.lower() == "no" :

            print("Look like you dont enjoy our service . We are improving! , Hope to see you back!")
            break

        else :

            print("Thanks for choosing us . We hope you will come back soon.")
            break

    print("IHATEPIZZA > " + kernel.respond(respond))



    
