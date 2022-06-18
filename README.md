# DeepCC v1.0

This release presents the source code and materials used for the experiments in our IEEE JSAC'21 paper: "Wanna Make Your TCP Scheme Great for Cellular Networks? Let Machines Do It for You!" <https://ieeexplore.ieee.org/document/9252929>.

Installation Guide
==================
Our patched Kernel is based on Kernel 4.13. That means you will need Ubuntu 14.04 or Ubuntu 16.04 to smoothly use our patched Kernel. As a side note, Ubuntu 16.04/14.04 are  based on 4.X Kernels while Ubuntu 18.04/0.04 are based on 5.X ones.

### Installing Required Tools

1. Install Mahimahi (http://mahimahi.mit.edu/#getting)

	```sh  
	sudo apt-get install build-essential git debhelper autotools-dev dh-autoreconf iptables protobuf-compiler libprotobuf-dev pkg-config libssl-dev dnsmasq-base ssl-cert libxcb-present-dev libcairo2-dev libpango1.0-dev iproute2 apache2-dev apache2-bin iptables dnsmasq-base gnuplot iproute2 apache2-api-20120211 libwww-perl
	git clone https://github.com/ravinet/mahimahi 
	cd mahimahi
	./autogen.sh && ./configure && make
	sudo make install
	sudo sysctl -w net.ipv4.ip_forward=1
	```

2. Install iperf3

	```sh
    sudo apt-get remove iperf3 libiperf0
    wget https://iperf.fr/download/ubuntu/libiperf0_3.1.3-1_amd64.deb
    wget https://iperf.fr/download/ubuntu/iperf3_3.1.3-1_amd64.deb
    sudo dpkg -i libiperf0_3.1.3-1_amd64.deb iperf3_3.1.3-1_amd64.deb
    rm libiperf0_3.1.3-1_amd64.deb iperf3_3.1.3-1_amd64.deb
	```

3. Install Python, pip, and virtual environment

Source code requires Python 3.4, 3.5, or 3.6. Python 3.x is generally installed by default on Ubuntu.

Check Python version
```
python3 --version
```

Install pip3 and virtual environment

```
sudo apt update
sudo apt install python3-pip
sudo pip3 install -U virtualenv
```

### Create Virtual environment

Create a new virtual environment by choosing a Python interpreter and make `~/venv` directory to hold it: 

```
sudo mkdir ~/venv
virtualenv --system-site-packages -p python3 ~/venv
```

4. Install Tensorflow (CPU)

Activate the virtual environment and install Tensorflow (TF version used for this work is 1.14).
```
source ~/venv/bin/activate
pip install --upgrade pip
pip install --upgrade tensorflow
```

5. Install sysv_ipc 1.0.0
```
pip install sysv_ipc
```

Verify Installation
```
python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"
```

To deactivate venv
```
(venv) $ deactivate
```

### Patching DeepCC Kernel: Install the prepared debian packages.

    cd linux
    sudo dpkg -i linux-image*
    sudo dpkg -i linux-header*
    sudo reboot 
 
### Verify the new kernel
Use the following command to make sure that new kernel is installed:

	uname -r

The output should be 4.13.1-0409. If not you need to bring the 4.13.1-0409 Kernel image on top of the grub list. For instance, you can use grub-customizer application. Install the grub-customizer using following:

   sudo add-apt-repository ppa:danielrichter2007/grub-customizer
   sudo apt-get update
   sudo apt-get install grub-customizer
   sudo grub-customizer

### Build DeepCC's Server-Client Apps
 To build the required applications, run the following:

    ./build.sh

### Cellular Traces 
    All gathered traces including some from prior work is located at the folder named traces.

### Use the learned model
    The trained models of TCP Cubic, BBR, Westwood, and Illinois used throughout the paper located in <models> folder.
    
    cd models/
    ./setitup.sh

### Run a Sample Test using the learned model

Here, we enable DeepCC for TCP Cubic and run a sample test over TMobile-LTE-driving trace:
    
	./evaluate.sh 8000 cubic  
    
Results will be generated automatically in rl-module/log/*
You can check out the summary of results in rl-module/log/results-deep-cubic*.tr file. 

