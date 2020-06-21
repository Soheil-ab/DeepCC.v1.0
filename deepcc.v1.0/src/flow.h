
#ifndef FLOW_H_
#define FLOW_H_

#include "common.h"
#include "util.h"


using namespace std;

struct sFlowinfo
{
	int sock;

	int flowid;

	int src_pod;
	int src_edg;
	int src_index;

	int dst_pod;
	int dst_edg;
	int dst_index;

//	unsigned int size;
	uint64_t size;
    uint64_t rem_size;
//    unsigned int rem_size;

	sFlowinfo& operator=(const sFlowinfo& other);
	void Copy(sFlowinfo);
	void Init();
};
enum EPrt_State
{
	eInit=0, 	//Check whether we need a Request
	eRQ,		//Waiting for the response
	eSTOP,		//Stop signal is received
	eGO			//Go signal is received
};

/**
 * Server-Side Flows!
 * Client has sent its request including data-size, {src,dst}
 */

class cFlow
{
public:
	cFlow();
	cFlow(sFlowinfo);
	void SetFlow(sFlowinfo info){flowinfo=info;};

	int SendData();

	void SetCtr(sockaddr_in addr,int ctr_sock){ctr_addr=addr;this->ctr_sock=ctr_sock;};
	void SetSrcDst(sockaddr_in src,sockaddr_in dst){src_addr=src;dst_addr=dst;};

public:
	bool qplus_enable;

	sockaddr_in ctr_addr;
	sockaddr_in src_addr;
	sockaddr_in dst_addr;

	int ctr_sock;

	pthread_mutex_t lock;
	sFlowinfo flowinfo;
	EPrt_State state;
	bool done;
	bool fin_msg_sent;
	bool startListening;
};

#endif /* FLOW_H_ */
