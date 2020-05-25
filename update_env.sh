#!/bin/bash
conda env update --name ideatransfer --file env.yml

# Note - the very first command you run to create the environment file is below.
# Take care that you install while specifying version number .... 
# Because the below command doesn't actually store the version numbers of libraries you installed! Just 
# what you typed.
# conda env export --from-history | grep -v prefix > env.yml
