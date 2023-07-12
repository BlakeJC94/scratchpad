# envy

A tiny little shell script to just do the damn python env  in as few keystrokes as possible

## Outline

With one command, I would lie the following to happen:
* If venv is active, deactivate it and exit
* Search for a `.venv` directory upload on the directory tree
    * Activate the first one that's found and exit
    * If not found, create a venv in the current directory, activate it and exit


