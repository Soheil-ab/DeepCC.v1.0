if [ $# -ne 2 ]
then
    echo "usage: $0 [server port num] [scheme: cubic westwood illi bbr ]"
    exit
fi
sudo sysctl -w -q net.ipv4.ip_forward=1
sudo sysctl -w -q net.ipv4.tcp_no_metrics_save=1
port=$1
scheme=$2

trace_index=25      # Check setup.sh for the valid trace indices.

latency=10 # Unidirectional Link Delay ==> minRTT>=2*latency (ms)
target=50  # ==>Overall application's Target delay. (ms)
report_period=20
./run-deep.sh $port ${trace_index} $target $report_period $latency 2 $scheme

