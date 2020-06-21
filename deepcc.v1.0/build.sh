g++ -pthread src/server.cc src/flow.cc -o rl-server
g++ src/client.c -o client

mv rl-server client rl-module/

