if [ $# -ne 3 ]
then
    echo "usage: $0 [learning from scratch: 1 , continue learning: 0] [server port num] [scheme: cubic westwood illi bbr ]"
    exit
fi

sudo sysctl -w -q net.ipv4.ip_forward=1
sudo sysctl -w -q net.ipv4.tcp_no_metrics_save=1
training=$1
port=$2
scheme=$3

latency=10 # Unidirectional Link Delay ==> minRTT>=2*latency (ms)
target=50  # ==>Overall application's Target delay. (ms)
report_period=20
trace_index=29
./run-deep.sh $port $trace_index $target $report_period $latency ${training} $scheme

