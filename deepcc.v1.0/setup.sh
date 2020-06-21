
##
##  TRACE INDEX:
##
## 0: TMobile-LTE-no-cross-times-static
## 1: TMobile-LTE-with-cross-times-static
## 2: trace-1552925703-home1-static
## 3: trace-1553093662-home3-static
## 4: trace-1553219061-home-static
## 5: trace-1553457194-ts-static
## 6: trace-1553458374-ts-static
## 7: trace-1553189663-ts-walking
## 8: trace-1553199718-ts-walking
## 9: trace-1553201037-ts-walking
## 10: trace-1553201711-ts-walking
## 11: trace-1553202253-ts-walking
## 12: trace-1553203600-ts-walking
## 13: trace-1553204803-ts-walking
## 14: trace-1553205967-ts-walking
## 15: trace-1553207521-ts-walking
## 16: trace-1553208852-ts-walking
## 17: trace-1553455408-ts-walking
## 18: trace-1553453943-ts-walking
## 19: trace-1553109898-bus
## 20: trace-1553114405-bus
## 21: trace-1553552192-bus
## 22: trace-1553555076-bus
## 23: trace-1552767958-taxi1
## 24: trace-1552768760-taxi3
## 25: TMobile-LTE-driving.down
## 26: Verizon-LTE-driving.down
## 27: ATT-LTE-driving.down
## 28: ATT-LTE-driving-2016.down
## 29: TMobile-LTE-driving.down     #Duration=36 hours

THIS_SCRIPT=$(cd "$(dirname "${BASH_SOURCE}")" ; pwd -P)
export SPROUT_MODEL_IN="$THIS_SCRIPT/alfalfa/src/examples/sprout.model"
export SPROUT_BT2="$THIS_SCRIPT/alfalfa/src/examples/sproutbt2"
export VERUS_SERVER="$THIS_SCRIPT/verus/src/verus_server"
export VERUS_CLIENT="$THIS_SCRIPT/verus/src/verus_client"

sudo sysctl -q net.ipv4.tcp_wmem="4096 32768 4194304" #Doubling the default value from 16384 to 32768
sudo sysctl -w -q net.ipv4.tcp_low_latency=1
sudo sysctl -w -q net.ipv4.tcp_autocorking=0
sudo sysctl -w -q net.ipv4.tcp_no_metrics_save=1
sudo sysctl -w -q net.ipv4.ip_forward=1

TRACES=("TMobile-LTE-no-cross-times-static" "TMobile-LTE-with-cross-times-static" "trace-1552925703-home1-static" "trace-1553093662-home3-static" "trace-1553219061-home-static" "trace-1553457194-ts-static" "trace-1553458374-ts-static" "trace-1553189663-ts-walking" "trace-1553199718-ts-walking" "trace-1553201037-ts-walking" "trace-1553201711-ts-walking"  "trace-1553202253-ts-walking" "trace-1553203600-ts-walking" "trace-1553204803-ts-walking" "trace-1553205967-ts-walking" "trace-1553207521-ts-walking" "trace-1553208852-ts-walking" "trace-1553455408-ts-walking" "trace-1553453943-ts-walking" "trace-1553109898-bus" "trace-1553114405-bus"  "trace-1553552192-bus" "trace-1553555076-bus" "trace-1552767958-taxi1" "trace-1552768760-taxi3" "TMobile-LTE-driving.down" "Verizon-LTE-driving.down" "ATT-LTE-driving.down" "ATT-LTE-driving-2016.down" "TMobile-LTE-driving.down")

duration=("943" "929" "299" "267" "1048" "914" "787" "889" "185" "171" "128" "391" "442" "188" "367" "259" "276" "312" "626" "923" "314" "895" "877" "170" "104" "474" "1363" "786" "120" "129600")

up="wired48"

parent=`pwd -P`
rl_dir="$parent/rl-module"
sudo killall rl-server rl-server-eval client python

