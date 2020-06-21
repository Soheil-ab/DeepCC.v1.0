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

#include "flow.h"
#include <pthread.h>
#include <sched.h>
#include <sys/types.h>        // needed for socket(), uint8_t, uint16_t, uint32_t
pthread_mutex_t lockit;
#include <unistd.h>
#include <math.h>
#include <time.h>
#define __STDC_FORMAT_MACROS
#include <inttypes.h>

//Shared Memory ==> Communication with RL-Module -----*
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
//-------------------------*
#define     OK_SIGNAL 99999
#define     TARGET_MARGIN   100      //100% ==> Full Target!
#define     TARGET_CHANGE_TIME 8  //Unit: minutes (Each TARGET_CHANGE_TIME minutes chagne the target)
#define     TARGET_CHANGE_STEP 25 
int shmid;
key_t key=123456;
char *shared_memory;
int shmid_rl;
key_t key_rl=12345;
char *shared_memory_rl;
int shmem_size=2048; //Shared Memory size: 2KBytes
//----------------------------------------------------*
typedef unsigned int u32;
typedef int s32;
typedef uint64_t  u64;
typedef int64_t  i64;

struct timeval tv_start,tv_start2;	//Start time (after three way handshake)
struct timeval tv_stamped,tv_stamped2;
uint64_t start_of_client;
struct timeval tv_end,tv_end2;		//End time
bool done=false;
bool check=false;
uint64_t setup_time;  //Flow Completion Time
char *downlink;
char *uplink;
char *log_file;
char *congestion;
int it=0;
bool got_message=0;
int codel=0;
int first_time=0;
int min_thr=2;
int qsize=100; //pkts
int flow_index=0;
int target_ratio=150;
u32 target=50; //50ms
u32 report_period=5;//5s
#define FLOW_NUM 1
int sock[FLOW_NUM];
int sock_for_cnt[FLOW_NUM];

struct tcp_deepcc_info {
    u32 min_rtt;      /* min-filtered RTT in uSec */
    u32 avg_urtt;     /* averaged RTT in uSec from the previous info request till now*/
    u32 cnt;          /* number of RTT samples uSed for averaging */
    unsigned long thr;          /*Bytes per second*/
    u32 thr_cnt;
    u32 cwnd;
    u32 pacing_rate;

    void init()
    {
        min_rtt=0;
        avg_urtt=0;
        cnt=0;
        thr=0;
        thr_cnt=0;
        cwnd=0;
        pacing_rate=0;
    }
    tcp_deepcc_info& operator =(const tcp_deepcc_info& a){
        this->min_rtt=a.min_rtt;
        this->avg_urtt=a.avg_urtt;
        this->cnt=a.cnt;
        this->thr=a.thr;
        this->thr_cnt=a.thr_cnt;
        this->cwnd=a.cwnd;
        this->pacing_rate=a.pacing_rate;
    }
}deepcc_info;

struct tcp_c2tcp_info {
    u32 c2tcp_min_rtt;      /* min-filtered RTT in uSec */
    u32 c2tcp_avg_urtt;     /* averaged RTT in uSec from the previous info request till now*/
    u32 c2tcp_cnt;          /* number of RTT samples uSed for averaging */
    unsigned long c2tcp_thr;          /*Bytes per second*/
    u32 c2tcp_thr_cnt;
    void init(){
        c2tcp_min_rtt=0;
        c2tcp_avg_urtt=0;
        c2tcp_cnt=0;
        c2tcp_thr=0;
        c2tcp_thr_cnt=0;
    }
    tcp_c2tcp_info& operator =(const tcp_c2tcp_info& a){
        this->c2tcp_min_rtt=a.c2tcp_min_rtt;
        this->c2tcp_avg_urtt=a.c2tcp_avg_urtt;
        this->c2tcp_cnt=a.c2tcp_cnt;
        this->c2tcp_thr=a.c2tcp_thr;
        this->c2tcp_thr_cnt=a.c2tcp_thr_cnt;

    }
}tcp_info;

struct tcp_c2tcp_info_signed {
    int c2tcp_min_rtt;      /* min-filtered RTT in uSec */
    int c2tcp_avg_urtt;     /* averaged RTT in uSec from the previous info request till now*/
    int c2tcp_cnt;          /* number of RTT samples uSed for averaging */
    signed long c2tcp_thr;          /*Bytes per second*/
    int c2tcp_thr_cnt;
    void init(){
        c2tcp_min_rtt=0;
        c2tcp_avg_urtt=0;
        c2tcp_cnt=0;
        c2tcp_thr=0;
        c2tcp_thr_cnt=0;
    }
    tcp_c2tcp_info_signed& operator =(const tcp_c2tcp_info_signed& a){
        this->c2tcp_min_rtt=a.c2tcp_min_rtt;
        this->c2tcp_avg_urtt=a.c2tcp_avg_urtt;
        this->c2tcp_cnt=a.c2tcp_cnt;
        this->c2tcp_thr=a.c2tcp_thr;
        this->c2tcp_thr_cnt=a.c2tcp_thr_cnt;
    }
};

struct tcp_c2tcp_info_signed info_diff(struct tcp_c2tcp_info a,struct tcp_c2tcp_info b){
    struct tcp_c2tcp_info_signed diff;
    diff.c2tcp_min_rtt=a.c2tcp_min_rtt-b.c2tcp_min_rtt;
    diff.c2tcp_avg_urtt=a.c2tcp_avg_urtt-b.c2tcp_avg_urtt;
    diff.c2tcp_cnt=a.c2tcp_cnt-b.c2tcp_cnt;
    diff.c2tcp_thr=a.c2tcp_thr-b.c2tcp_thr;
    diff.c2tcp_thr_cnt=a.c2tcp_thr_cnt-b.c2tcp_thr_cnt;
    return diff;
}

struct tcp_c2tcp_info_signed info_diff_2nd(struct tcp_c2tcp_info_signed a,struct tcp_c2tcp_info_signed b){
    struct tcp_c2tcp_info_signed diff;
    diff.c2tcp_min_rtt=a.c2tcp_min_rtt-b.c2tcp_min_rtt;
    diff.c2tcp_avg_urtt=a.c2tcp_avg_urtt-b.c2tcp_avg_urtt;
    diff.c2tcp_cnt=a.c2tcp_cnt-b.c2tcp_cnt;
    diff.c2tcp_thr=a.c2tcp_thr-b.c2tcp_thr;
    diff.c2tcp_thr_cnt=a.c2tcp_thr_cnt-b.c2tcp_thr_cnt;
    return diff;
}
                    
struct sTrace
{
    double time;
    double bw;
    double minRtt;
};
struct sInfo
{
    sTrace *trace;
    int sock;
    int num_lines;
};
int delay_ms;
int client_port;
sTrace *trace;


#define TARGET_RATIO_MIN 100
#define TARGET_RATIO_MAX 1000

#define TCP_CWND_CLAMP 42
#define TCP_CWND 43
#define TCP_DEEPCC_ENABLE 44
#define TCP_CWND_CAP 45
#define TCP_DEEPCC_INFO 46 /* Get          Congestion Control (optional) DeepCC info */

/*C2TCP*/
#define  TCP_C2TCP_ENABLE 50
#define  TCP_C2TCP_ALPHA_INTERVAL 51
#define  TCP_C2TCP_ALPHA_SETPOINT 52
#define  TCP_C2TCP_ALPHA 53
#define  TCP_C2TCP_X 54
#define TCP_CC_INFO 26  /* Get Congestion Control (optional)            info */


uint64_t raw_timestamp( void )
{
    struct timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    uint64_t us = ts.tv_nsec / 1000;
    us += (uint64_t)ts.tv_sec * 1000000;
    return us;
}
uint64_t timestamp_begin(bool set)
{
        static uint64_t start;
        if(set)
            start = raw_timestamp();
        return start;
}
uint64_t timestamp_end( void )
{
        return raw_timestamp() - timestamp_begin(0);
}

uint64_t initial_timestamp( void )
{
        static uint64_t initial_value = raw_timestamp();
        return initial_value;
}

uint64_t timestamp( void )
{
        return raw_timestamp() - initial_timestamp();
}

//Start server
void start_server(int flow_num, int client_port);

//thread functions
void* DataThread(void*);
void* CntThread(void*);
void* TimerThread(void*);

//Print usage information
void usage();

int get_deepcc_info(int sk, struct tcp_deepcc_info *info)
{
    int tcp_info_length = sizeof(*info);

    return getsockopt( sk, SOL_TCP, TCP_DEEPCC_INFO, (void *)info, (socklen_t *)&tcp_info_length );
};

int get_info(int sk, struct tcp_c2tcp_info *info)
{
    int tcp_info_length = sizeof(*info);

    return getsockopt( sk, SOL_TCP, TCP_CC_INFO, (void *)info, (socklen_t *)&tcp_info_length );
};

void handler(int sig) {
    void *array[10];
    size_t size;
    DBGMARK(DBGSERVER,2,"=============================================================== Start\n");
    size = backtrace(array, 20);
    fprintf(stderr, "We got signal %d:\n", sig);
    DBGMARK(DBGSERVER,2,"=============================================================== End\n");
    shmdt(shared_memory);
    shmctl(shmid, IPC_RMID, NULL);
    shmdt(shared_memory_rl);
    shmctl(shmid_rl, IPC_RMID, NULL);
    exit(1);
}

