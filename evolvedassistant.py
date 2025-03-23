from rag.assistant import Assistant

if __name__ == "__main__":

    assistant = Assistant()

    for chunk in assistant.get_graph().stream({"query":
                                                  #"What is my username ?"
                                                  #"Create a folder named test-folder and containing 10 files with random names"
                                                  #"In my home directory create a folder named letters with 5 folders inside with the 5 letters of the alphabet for each of their name"
                                                  #"Give me a summary of my network configuration in /etc/network"
                                                  #"Give me the list of all files and folders in my home directory"
                                                  "Create a docker compose file containing a postgres db container, you have to create it in the 'docker-test' folder"
                                                  , "path": "/home/octave/"}, {"recursion_limit": 100}, subgraphs=True):
        print(chunk)