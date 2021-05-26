g++ -pthread src/server.cc src/flow.cc -o rl-server
g++ src/client.c -o client

mv rl-server client rl-module/

# Make the log directory for the experiments
mkdir -p rl-module/log
