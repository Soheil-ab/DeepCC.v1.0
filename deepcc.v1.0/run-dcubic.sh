#!/bin/bash

if [ $# != 11 ]
then
    echo -e "usage:$0 latency (one-way delay) port trace_id it qsize target initial_alpha period training scheme[ cubic , vegas , west , illi , yeah , veno, scal , htcp , cdg , hybla ,  ] result_prefix"
    echo "$@"
    echo "$#"
    exit
fi
sudo killall -s9 server client tcpdump rl-server
sudo killall -s15 python
#Remove Shared Memories if they have not been destroyed yet.
#ipcrm -M 123456
#ipcrm -M 12345
#ipcs -m

#One way delay (do not confuse it with minRTT)
latency=$1

port=$2
i=$3;
it=$4
qsize=$5
target=$6
initial_alpha=$7
period=$8
training=${9}
scheme=${10}
prefix=${11}
x=100

source setup.sh
sudo su <<EOF
echo "1" > /proc/sys/net/ipv4/tcp_deepcc
EOF

if [ "$scheme" == "cubic" ] || [ "$scheme" == "bbr" ] || [ "$scheme" == "westwood" ] || [ "$scheme" == "illinois" ]
then
    a=""
else
    echo "-------------- Pls give a valid scheme! ------------- ($scheme ?)"
    exit 0
fi

cd ${rl_dir}
echo "Running Deep-$scheme: ${TRACES[$i]}"
scheme_des="deep-$scheme-$latency-$target-$period"
log="deep-$scheme-${TRACES[$i]}-$latency-${qsize}-${target}-${initial_alpha}-${period}-${it}"
sudo ./rl-server $latency $port ${TRACES[$i]} $up ${log} $target $initial_alpha ${qsize} ${period} 2 ${training} $scheme &
pid=$!
echo $pid
sleep 4;
echo "Will be done in ${duration[$i]} seconds ..."
sleep ${duration[$i]}
sudo kill $pid
sudo killall rl-server mm-link client
sudo killall -s15 python
sleep 10
echo "Experiment is finished."
if [ ${training} -eq 2 ] || [ ${training} -eq 4 ]
then
    echo "Doing analysis ..."
    out="$prefix-${log}.tr"
    echo $log >> log/$out
    ./mm-thr 500 log/down-${log} 1>log/down-$log.svg 2>res_tmp
    cat res_tmp >>log/$out
    echo "------------------------------" >> log/$out
fi
echo "Done with analysis."
cd ..

