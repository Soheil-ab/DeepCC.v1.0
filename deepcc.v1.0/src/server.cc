/*////////////////////////////////////////////////////////////////////////////////
  MIT License
  Copyright (c) 2019 Soheil Abbasloo

  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/////////////////////////////////////////////////////////////////////////////////

#define DBGSERVER 0
#include <cstdlib>
#include "define.h"
#define MAX_CWND 40000
#define MIN_CWND 4
#define MIN_RTT_DURING_TRAINING 21

int main(int argc, char **argv)
{
    DBGPRINT(DBGSERVER,4,"Main\n");
    if(argc!=13)
	{
        DBGERROR("argc:%d\n",argc);
        for(int i=0;i<argc;i++)
        	DBGERROR("argv[%d]:%s\n",i,argv[i]);
		usage();
		return 0;
	}
    
    srand(raw_timestamp());

	signal(SIGSEGV, handler);   // install our handler
	signal(SIGTERM, handler);   // install our handler
	signal(SIGABRT, handler);   // install our handler
	signal(SIGFPE, handler);   // install our handler
    signal(SIGKILL,handler);   // install our handler
    int flow_num;
	const char* cnt_ip;
    cnt_ip="10.10.10.10";
	int cnt_port;

	bool qplus_enable;
	flow_num=FLOW_NUM;
	//atoi(argv[1]);
	delay_ms=atoi(argv[1]);
	client_port=atoi(argv[2]);
	qplus_enable=0;
    downlink=argv[3];
    uplink=argv[4];
    log_file=argv[5];
    target_ratio=atoi(argv[7]);
    target=atoi(argv[6]);
    qsize=atoi(argv[8]);
    report_period=atoi(argv[9]);
	codel=atoi(argv[10]);
   	first_time=atoi(argv[11]);
    congestion=argv[12];
    start_server(flow_num, client_port);
	DBGMARK(DBGSERVER,5,"DONE!\n");
    shmdt(shared_memory);
    shmctl(shmid, IPC_RMID, NULL);
    shmdt(shared_memory_rl);
    shmctl(shmid_rl, IPC_RMID, NULL);
    return 0;
}

void usage()
{
	DBGMARK(0,0,"./server [Delay(ms)] [port] [DL-trace] [UP-trace] [log] [Target] [Initial Alpha] [qsize in pkts] [Report Period: 1 sec] [AQM:2 FIFO] [First Time: 1=yes, 0=no] [tcp:cubic westwood illinois bbr ...]\n");
}

void start_server(int flow_num, int client_port)
{
	cFlow *flows;
    int num_lines=0;
	bool qplus_enable=0;
	FILE* filep;
	sInfo *info;
	info = new sInfo;
	char line[4096];
	int msec = 0, trigger = 10; /* 10ms */
	clock_t before = clock();
	flows = new cFlow[flow_num];

	int cnt_port=PORT_CTR;
	if(flows==NULL)
	{
		DBGMARK(0,0,"flow generation failed\n");
		return;
	}

	//threads
	pthread_t data_thread;
	pthread_t cnt_thread;
	pthread_t timer_thread;

	//Server address
	struct sockaddr_in server_addr[FLOW_NUM];
	//Client address
	struct sockaddr_in client_addr[FLOW_NUM];
	//Controller address
	struct sockaddr_in ctr_addr;
    for(int i=0;i<FLOW_NUM;i++)
    {
        memset(&server_addr[i],0,sizeof(server_addr[i]));
        //IP protocol
        server_addr[i].sin_family=AF_INET;
        //Listen on "0.0.0.0" (Any IP address of this host)
        server_addr[i].sin_addr.s_addr=INADDR_ANY;
        //Specify port number
        server_addr[i].sin_port=htons(client_port+i);

        //Init socket
        if((sock[i]=socket(PF_INET,SOCK_STREAM,0))<0)
        {
            DBGMARK(0,0,"sockopt: %s\n",strerror(errno));
            return;
        }

        int reuse = 1;
        if (setsockopt(sock[i], SOL_SOCKET, SO_REUSEADDR, (const char*)&reuse, sizeof(reuse)) < 0)
            perror("setsockopt(SO_REUSEADDR) failed");
        //Bind socket on IP:Port
        if(bind(sock[i],(struct sockaddr *)&server_addr[i],sizeof(struct sockaddr))<0)
        {
            DBGMARK(0,0,"bind error srv_ctr_ip: 000000: %s\n",strerror(errno));
            close(sock[i]);
            return;
        }
        if (congestion) 
        {
            if (setsockopt(sock[i], IPPROTO_TCP, TCP_CONGESTION, congestion, strlen(congestion)) < 0) 
            {
                DBGMARK(0,0,"TCP congestion doesn't exist: %s\n",strerror(errno));
                return;
            } 
        }
    }
    char container_cmd[500];
    sprintf(container_cmd,"sudo ./client $MAHIMAHI_BASE 1 %d",client_port);
    for(int i=1;i<FLOW_NUM;i++)
    {
    	sprintf(container_cmd,"%s & sleep 5;sudo ./client $MAHIMAHI_BASE 1 %d",container_cmd,client_port+i);
    }
    char cmd[1000];
    char final_cmd[1000];
    sprintf(cmd, "sudo -u `logname` mm-delay %d mm-link ../traces/%s ../traces/%s --downlink-log=log/down-%s --uplink-queue=droptail --uplink-queue-args=\"packets=%d\" --downlink-queue=droptail --downlink-queue-args=\"packets=%d\" -- sh -c \'%s\' &",delay_ms,uplink,downlink,log_file,qsize,qsize,container_cmd);   
    sprintf(final_cmd,"%s",cmd);
    info->trace=trace;
    info->num_lines=num_lines;
    /**
     *Setup Shared Memory
     */ 
    key=(key_t) (rand()%1000000+1);
    key_rl=(key_t) (rand()%1000000+1);
    // Setup shared memory
    if ((shmid = shmget(key, shmem_size, IPC_CREAT | 0666)) < 0)
    {
        printf("Error getting shared memory id");
        return;
    }
        // Attached shared memory
    if ((shared_memory = (char*)shmat(shmid, NULL, 0)) == (char *) -1)
    {
        printf("Error attaching shared memory id");
        return;
    }
    // Setup shared memory
    if ((shmid_rl = shmget(key_rl, shmem_size, IPC_CREAT | 0666)) < 0)
    {
        printf("Error getting shared memory id");
        return;
    }
    // Attached shared memory
    if ((shared_memory_rl = (char*)shmat(shmid_rl, NULL, 0)) == (char *) -1)
    {
        printf("Error attaching shared memory id");
        return;
    }
    
    if (first_time==1){
        sprintf(cmd,"sudo /home/`logname`/venv/bin/python drl_agent.py --target=%.7f --tb_interval=1 --scheme=%s --train_dir=\".\" --mem_r=%d --mem_w=%d &",
                (double)target,congestion,(int)key,(int)key_rl);
        DBGPRINT(0,0,"Starting RL Module (Without load) ...\n");
    }
    else if (first_time==2 || first_time==4){
        sprintf(cmd,"sudo /home/`logname`/venv/bin/python drl_agent.py --target=%.7f --tb_interval=1 --load=1 --eval --scheme=%s --train_dir=\".\" --mem_r=%d --mem_w=%d &",
                (double)target,congestion,(int)key,(int)key_rl);
        DBGPRINT(0,0,"Starting RL Module (No learning) ...\n");
    }
    else
    {
        sprintf(cmd,"sudo /home/`logname`/venv/bin/python drl_agent.py --target=%.7f --load=1 --tb_interval=1 --scheme=%s --train_dir=\".\" --mem_r=%d --mem_w=%d &",
                (double)target,congestion,(int)key,(int)key_rl);
        DBGPRINT(0,0,"Starting RL Module (With load) ...\n");
    }
 
    initial_timestamp();
    system(cmd);
    //Wait to get OK signal (alpha=OK_SIGNAL)
    bool got_ready_signal_from_rl=false;
    int signal;
    char *num;
    char*alpha;
    char *save_ptr;
    while(!got_ready_signal_from_rl)
    {
        //Get alpha from RL-Module
        num=strtok_r(shared_memory_rl," ",&save_ptr);
        alpha=strtok_r(NULL," ",&save_ptr);
        if(num!=NULL && alpha!=NULL)
        {
           signal=atoi(alpha);      
           if(signal==OK_SIGNAL)
           {
              got_ready_signal_from_rl=true;
           }
           else{
               usleep(10000);
           }
        }
        else{
           usleep(10000);
        }
    }
    DBGPRINT(0,0,"RL Module is Ready. Let's Start ...\n\n");
    
    //Now its time to start the server-client app and tune C2TCP socket.
    system(final_cmd);
    
    //Start listen
	//The maximum number of concurrent connections is 10
	for(int i=0;i<FLOW_NUM;i++)
    {
        listen(sock[i],10);
    }
	int sin_size=sizeof(struct sockaddr_in);
	while(flow_index<flow_num)
	{
		int value=accept(sock[flow_index],(struct sockaddr *)&client_addr[flow_index],(socklen_t*)&sin_size);
		if(value<0)
		{
			perror("accept error\n");
			DBGMARK(0,0,"sockopt: %s\n",strerror(errno));
			DBGMARK(0,0,"sock::%d, index:%d\n",sock[flow_index],flow_index);
            close(sock[flow_index]);
			return;
		}
		sock_for_cnt[flow_index]=value;
		flows[flow_index].flowinfo.sock=value;
		flows[flow_index].dst_addr=client_addr[flow_index];
		if(pthread_create(&data_thread, NULL , DataThread, (void*)&flows[flow_index]) < 0)
		{
			perror("could not create thread\n");
			close(sock[flow_index]);
			return;
		}
                  
            if (flow_index==0)
            {
                if(pthread_create(&cnt_thread, NULL , CntThread, (void*)info) < 0)
                {
                    perror("could not create control thread\n");
                    close(sock[flow_index]);
                    return;
                }
                if(pthread_create(&timer_thread, NULL , TimerThread, (void*)info) < 0)
                {
                    perror("could not create timer thread\n");
                    close(sock[flow_index]);
                    return;
                }
            }
        DBGPRINT(0,0,"Server is Connected to the client...\n");
            flow_index++;
	}
    pthread_join(data_thread, NULL);
}
void* TimerThread(void* information)
{
    int index=1;
    int pre_index=1;
    int target_array[3];
    target_array[0]=target/2;
    target_array[1]=target;
    target_array[2]=(target*3)/2;
#ifdef CHANGE_TARGET
    if (first_time!=2)
    {
        while(true)
        {
            sleep(TARGET_CHANGE_TIME*60);
            pre_index=index;
            index=(index+1)%3;
            target = target_array[index];
            DBGPRINT(DBGSERVER,0,"Target changed from %d to %d \n",target_array[pre_index],target_array[index]);
        }
    }
#endif
    return((void *)0);
}
void* CntThread(void* information)
{
//    printf("testing\n");
    struct sched_param param;
    param.__sched_priority=sched_get_priority_max(SCHED_RR);
    int policy=SCHED_RR;
    int s = pthread_setschedparam(pthread_self(), policy, &param);
    if (s!=0)
    {
        DBGERROR("Cannot set priority (%d) for the Main: %s\n",param.__sched_priority,strerror(errno));
    }

    s = pthread_getschedparam(pthread_self(),&policy,&param);
    if (s!=0)
    {
        DBGERROR("Cannot get priority for the Data thread: %s\n",strerror(errno));
    }
    uint64_t fct_=start_of_client-initial_timestamp();
    sInfo* info = (sInfo*)information;
	int val1,pre_val1=0,val3=1,val4=0,val5=0,val6=0;
	int val2,pre_val2=0;
    int64_t tmp;
	int ret1;
	int ret2;
    bool strated=0;
	socklen_t optlen;
	optlen = sizeof val1;
	double preTime=0;
	double delta=0;
    int64_t offset=0;
    double bias_time=0;
    double min_rtt_=0.0;
	int reuse = 1;
    int pre_id=9230;
    int pre_id_tmp=0;
    int msg_id=657;
    bool got_alpha=false;
    for(int i=0;i<FLOW_NUM;i++)
    {
        if (setsockopt(sock_for_cnt[i], IPPROTO_TCP, TCP_NODELAY, &reuse, sizeof(reuse)) < 0)
        {
            DBGMARK(0,0,"ERROR: set TCP_NODELAY option %s\n",strerror(errno));
            return((void *)0);
        }
    }
    char message[1000];
    char *num;
    char*alpha;
    char*save_ptr;
    int got_no_zero=0;
    uint64_t t0,t1;
    t0=timestamp();
    //Time to start the Logic
    struct tcp_deepcc_info tcp_info_pre;
    tcp_info_pre.init();
    while(true)  
	{
       for(int i=0;i<flow_index;i++)
       {
           got_no_zero=0;
           usleep(report_period*1000);
           while(!got_no_zero)
           {
                ret1= get_deepcc_info(sock_for_cnt[i],&deepcc_info);
                if(ret1<0)
                {
                    DBGMARK(0,0,"Error: setsockopt: for index:%d flow_index:%d TCP_C2TCP ... %s (ret1:%d)\n",i,flow_index,strerror(errno),ret1);
                    return((void *)0);
                }
                if(deepcc_info.avg_urtt>0)
                {
                    t1=timestamp();
                    double time_delta=(double)(t1-t0)/1000000.0;
                    min_rtt_=(double)(deepcc_info.min_rtt/1000.0);
//                    report_period=min_rtt_;
//                    if(report_period<20)
//                        report_period=20;
                    double delay=(double)deepcc_info.avg_urtt/1000.0;
                    sprintf(message,"%d %.7f %.7f %.7f %.7f %.7f %7.f %7.f",
                            msg_id,delay-min_rtt_+MIN_RTT_DURING_TRAINING,(double)deepcc_info.thr,(double)deepcc_info.cnt,
                            (double)time_delta,(double)target+MIN_RTT_DURING_TRAINING-min_rtt_,(double)deepcc_info.cwnd, (double)deepcc_info.pacing_rate);
//                    sprintf(message,"%d %.7f %.7f %.7f %.7f %.7f %7.f %7.f",
//                            msg_id,delay,(double)deepcc_info.thr,(double)deepcc_info.cnt,
//                            (double)time_delta,(double)target,(double)deepcc_info.cwnd, (double)deepcc_info.pacing_rate);
                    memcpy(shared_memory,message,sizeof(message));
                    msg_id=(msg_id+1)%1000;
//                    DBGPRINT(DBGSERVER,0,"%.7f %.7f %.7f\n",min_rtt_,delay, delay-min_rtt_+20);
                    DBGPRINT(DBGSERVER,1,"%s\n",message);
                    got_no_zero=1;
                    tcp_info_pre=deepcc_info;
                    t0=timestamp();
                }
                else
                {
                    usleep(report_period*100);
                }
           }
        //
        got_alpha=false;
        int error_cnt=0;
        int error2_cnt=0;
        while(!got_alpha)
        { 
           //Get alpha from RL-Module
           num=strtok_r(shared_memory_rl," ",&save_ptr);
           alpha=strtok_r(NULL," ",&save_ptr);
           if(num!=NULL && alpha!=NULL)
           {
               pre_id_tmp=atoi(num);
               target_ratio=atoi(alpha);
               if(pre_id!=pre_id_tmp /*&& target_ratio!=OK_SIGNAL*/)
               {
                  got_alpha=true; 
                  pre_id=pre_id_tmp; 
                  target_ratio=atoi(alpha)*deepcc_info.cwnd/100;
                  if (target_ratio<MIN_CWND)
                      target_ratio=MIN_CWND;
                  if (target_ratio>MAX_CWND)
                      target_ratio=MAX_CWND;
                  DBGPRINT(DBGSERVER,1,"------------old.Cwnd: %d New.Cwnd: %d\n",deepcc_info.cwnd,target_ratio);
                  ret1 = setsockopt(sock_for_cnt[i], IPPROTO_TCP,TCP_CWND_CAP, &target_ratio, sizeof(target_ratio));
                  if(ret1<0)
                  {
                      DBGPRINT(0,0,"Error: setsockopt: for index:%d flow_index:%d ... %s (ret1:%d)\n",i,flow_index,strerror(errno),ret1);
                      return((void *)0);
                  }
                  error_cnt=0;
               }
               else{
                   if (error_cnt>1000)
                   {
                       DBGPRINT(DBGSERVER,2,"still no new value id:%d prev_id:%d\n",pre_id_tmp,pre_id);
                       error_cnt=0;
                   }
                   error_cnt++;
                   usleep(1000);
               }
               error2_cnt=0;
           }
           else{
                if (error2_cnt>1000)
                {
                    DBGPRINT(DBGSERVER,2,"****************** got null values **********************\n");
                    got_alpha=true; 
                    error2_cnt=0;
                }
                else{ 
                    error2_cnt++;
                    usleep(1000);
                }
           }
        }
     
       }
    }
    shmdt(shared_memory);
    shmctl(shmid, IPC_RMID, NULL);
    shmdt(shared_memory_rl);
    shmctl(shmid_rl, IPC_RMID, NULL);
    return((void *)0);
}
void* DataThread(void* info)
{
	struct sched_param param;
    param.__sched_priority=sched_get_priority_max(SCHED_RR);
    int policy=SCHED_RR;
    int s = pthread_setschedparam(pthread_self(), policy, &param);
    if (s!=0)
    {
        DBGERROR("Cannot set priority (%d) for the Main: %s\n",param.__sched_priority,strerror(errno));
    }

    s = pthread_getschedparam(pthread_self(),&policy,&param);
    if (s!=0)
    {
        DBGERROR("Cannot get priority for the Data thread: %s\n",strerror(errno));
    }
    pthread_t send_msg_thread;

	cFlow* flow = (cFlow*)info;
	int sock_local = flow->flowinfo.sock;
	char* src_ip;
	char write_message[BUFSIZ+1];
	char read_message[1024]={0};
	int len;
	char *savePtr;
	char* dst_addr;
	u64 loop;
	u64  remaining_size;

	memset(write_message,1,BUFSIZ);
	write_message[BUFSIZ]='\0';
	/**
	 * Get the RQ from client : {src_add} {flowid} {size} {dst_add}
	 */
	len=recv(sock_local,read_message,1024,0);
	if(len<=0)
	{
		DBGMARK(DBGSERVER,1,"recv failed! \n");
		close(sock_local);
		return 0;
	}
	/**
	 * For Now: we send the src IP in the RQ to!
	 */
	src_ip=strtok_r(read_message," ",&savePtr);
	if(src_ip==NULL)
	{
		//discard message:
		DBGMARK(DBGSERVER,1,"id: %d discarding this message:%s \n",flow->flowinfo.flowid,savePtr);
		close(sock_local);
		return 0;
	}
	char * isstr = strtok_r(NULL," ",&savePtr);
	if(isstr==NULL)
	{
		//discard message:
		DBGMARK(DBGSERVER,1,"id: %d discarding this message:%s \n",flow->flowinfo.flowid,savePtr);
		close(sock_local);
		return 0;
	}
	flow->flowinfo.flowid=atoi(isstr);
	char* size_=strtok_r(NULL," ",&savePtr);
	flow->flowinfo.size=1024*atoi(size_);
    DBGPRINT(DBGSERVER,4,"%s\n",size_);
	dst_addr=strtok_r(NULL," ",&savePtr);
	if(dst_addr==NULL)
	{
		//discard message:
		DBGMARK(DBGSERVER,1,"id: %d discarding this message:%s \n",flow->flowinfo.flowid,savePtr);
		close(sock_local);
		return 0;
	}
	char* time_s_=strtok_r(NULL," ",&savePtr);
    char *endptr;
    start_of_client=strtoimax(time_s_,&endptr,10);
	got_message=1;
    DBGPRINT(DBGSERVER,2,"Got message: %" PRIu64 " us\n",timestamp());
    flow->flowinfo.rem_size=flow->flowinfo.size;
    DBGPRINT(DBGSERVER,2,"time_rcv:%" PRIu64 " get:%s\n",start_of_client,time_s_);

	//Get detailed address
	strtok_r(src_ip,".",&savePtr);
	if(dst_addr==NULL)
	{
		//discard message:
		DBGMARK(DBGSERVER,1,"id: %d discarding this message:%s \n",flow->flowinfo.flowid,savePtr);
		close(sock_local);
		return 0;
	}

	//////////////////////////////////////////////////////////////////////////////
	char query[150];	//query=data_size+' '+deadline+' '+agnostic
	char strTmp[150];
	char strTmp2[150];

	int sockfd;	//Socket
	//////////////////////////////////////////////////////////////////////////////

	//Calculate loops. In each loop, we can send BUFSIZ (8192) bytes of data
	loop=flow->flowinfo.size/BUFSIZ*1024;
	//Calculate remaining size to be sent
	remaining_size=flow->flowinfo.size*1024-loop*BUFSIZ;
	//Send data with 8192 bytes each loop
    //DBGPRINT(DBGSERVER,5,"size:%" PRId64 "\trem_size:%u,loop:%" PRId64 "\n",flow->flowinfo.size*1024,remaining_size,loop);
	DBGPRINT(0,0,"Server is sending the traffic ...\n");

   // for(u64 i=0;i<loop;i++)
	while(true)
    {
		len=strlen(write_message);
		while(len>0)
		{
			DBGMARK(DBGSERVER,5,"++++++\n");
			len-=send(sock_local,write_message,strlen(write_message),0);
		    usleep(50);         
            DBGMARK(DBGSERVER,5,"      ------\n");
		}
        usleep(100);
	}
	//Send remaining data
	memset(write_message,1,BUFSIZ);
	write_message[remaining_size+1]='\0';
	len=strlen(write_message);
	DBGMARK(DBGSERVER,3,"remaining starts\n");
	while(len>0)
	{
		len-=send(sock_local,write_message,strlen(write_message),0);
		DBGMARK(DBGSERVER,3,"*******************\n");
	}
	DBGMARK(DBGSERVER,3,"remaining finished\n");
	flow->flowinfo.rem_size=0;
    done=true;
    DBGPRINT(DBGSERVER,1,"done=true\n");
    close(sock_local);
    DBGPRINT(DBGSERVER,1,"done\n");
	return((void *)0);
}
