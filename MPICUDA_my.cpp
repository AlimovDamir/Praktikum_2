#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm> 
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <sys/time.h>

#define Print
#define TRUE  ((int) 1)
#define FALSE ((int) 0)

bool cuda_setka(cudaDeviceProp dev, cudaStream_t cudaStreams, double* const x, int n, int Coords, int k, double h);

bool cuda_gran(cudaDeviceProp dev, cudaStream_t cudaStreams, double* const x, int n, double a, const double* const b, int c1, int c2);

bool cuda_rpart(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* const rhs, int n0, int n1, const double* const x, const double* const y);

bool cuda_prisv(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* const x, const double* const y, int n0, int n1, int o1, int o2, int o3, int o4, double a, double b);

bool cuda_sendm(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* x, int n, const double* b, int c1, int c2);

bool cuda_pint(cudaDeviceProp dev, cudaStream_t* cudaStreams, const double* const rhs, int n0, int n1, double* const x, int n00, int o2, int o3);

bool cuda_recvm(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* x, int n,const double* b, int c1, int c2, int c3);

bool cuda_left(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* x, const double* const y, int n0, int n1, int o1, int o2, int o3, int o4, int n00, double hx, double hy);

bool cuda_zero(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* rhs, int n0, int n1);

double cuda_scal(cudaDeviceProp dev, cudaStream_t* cudaStreams, const double * const v1, const double * const v2,const int n0,const int n1,const double hx,const double hy, MPI_Comm com, bool* fle);

bool cuda_crit(cudaDeviceProp dev, cudaStream_t* cudaStreams, const double * v1, const double * v2, const int n0, const int n1, const double hx, const double hy, MPI_Comm com, bool* fle);

int IsPower(int Number)
// the function returns log_{2}(Number) if it is integer. If not it returns (-1). 
{
    unsigned int M;
    int p;
    
    if(Number <= 0)
        return(-1);
        
    M = Number; p = 0;
    while(M % 2 == 0)
    {
        ++p;
        M = M >> 1;
    }
    if((M >> 1) != 0)
        return(-1);
    else
        return(p);
    
}

int SplitFunction(int N0, int N1, int p)
// This is the splitting procedure of proc. number p. The integer p0
// is calculated such that abs(N0/p0 - N1/(p-p0)) --> min.
{
    float n0, n1;
    int p0, i;
    
    n0 = (float) N0; n1 = (float) N1;
    p0 = 0;
    
    for(i = 0; i < p; i++)
        if(n0 > n1)
        {
            n0 = n0 / 2.0;
            ++p0;
        }
        else
            n1 = n1 / 2.0;
    
    return(p0);
}

bool LeftPart(double *sbuf_up, double *sbuf_down, double *sbuf_right, double *sbuf_left, double *rbuf_up, double *rbuf_down, double *rbuf_right, double *rbuf_left, cudaDeviceProp dev, cudaStream_t* cudaStreams, double * rhs, double * res, int n0, int n1, double hx, double hy, int left, int right, int down, int up, MPI_Comm com, int nei)
{
	int send = 0;
	int recv = 0;
	int ret;
	int o1 = 0;
	int o2 = 0;
	int o3 = 0;
	int o4 = 0;
	int n00;
	int n11;
	double *sbuf_uph;
	double *sbuf_downh;
	double *sbuf_righth;
	double *sbuf_lefth;
	
	double *rbuf_uph;
	double *rbuf_downh;
	double *rbuf_righth;
	double *rbuf_lefth;
	bool fle = false;
	MPI_Request *sreq;
	MPI_Request *rreq;
    n00 = n0;
    n11 = n1;	
	
	if (nei != 0)
	{
		sbuf_uph = (double *)malloc(n0*sizeof(double));
		sbuf_downh = (double *)malloc(n0*sizeof(double));
		sbuf_righth = (double *)malloc(n1*sizeof(double));
		sbuf_lefth = (double *)malloc(n1*sizeof(double));
		rbuf_uph = (double *)malloc(n0*sizeof(double));
		rbuf_downh = (double *)malloc(n0*sizeof(double));
		rbuf_righth = (double *)malloc(n1*sizeof(double));
		rbuf_lefth = (double *)malloc(n1*sizeof(double));
		sreq = (MPI_Request *)malloc(4*sizeof(MPI_Request));
		rreq = (MPI_Request *)malloc(4*sizeof(MPI_Request));
		if(right >= 0)
		{
			fle = fle || cuda_sendm(dev, cudaStreams, sbuf_right, n1, rhs, n0, n0-1);
			cudaMemcpy(sbuf_righth, sbuf_right, n1*sizeof(double), cudaMemcpyDeviceToHost);
			ret = MPI_Isend(sbuf_righth, n1, MPI_DOUBLE, right, 1, com, &(sreq[send]));
			send = send + 1;
			n00 = n00 + 1;
			o1 = 1;
		}
		
		if(left >= 0)
		{
			fle = fle || cuda_sendm(dev, cudaStreams, sbuf_left, n1, rhs, n0, 0);
			cudaMemcpy(sbuf_lefth, sbuf_left, n1*sizeof(double), cudaMemcpyDeviceToHost);
			ret = MPI_Isend(sbuf_lefth, n1, MPI_DOUBLE, left, 2, com, &(sreq[send]));
			send = send + 1;
			n00 = n00 + 1;
			o2 = 1;
		}
		
		if(down >= 0)
		{
			fle = fle || cuda_sendm(dev, cudaStreams, sbuf_down, n0, rhs, 1, 0);
			cudaMemcpy(sbuf_downh, sbuf_down, n0*sizeof(double), cudaMemcpyDeviceToHost);
			ret = MPI_Isend(sbuf_downh, n0, MPI_DOUBLE, down, 3, com, &(sreq[send]));
			send = send + 1;
			n11 = n11 + 1;
			o3 = 1;
		}
		
		if(up >= 0)
		{
			fle = fle || cuda_sendm(dev, cudaStreams, sbuf_up, n0, rhs, 1, n0*(n1-1));
			cudaMemcpy(sbuf_uph, sbuf_up, n0*sizeof(double), cudaMemcpyDeviceToHost);
			ret = MPI_Isend(sbuf_uph, n0, MPI_DOUBLE, up, 4, com, &(sreq[send]));
			send = send + 1;
			n11 = n11 + 1;
			o4 = 1;
		}

		if(right >= 0)
		{
			ret = MPI_Irecv(rbuf_righth, n1, MPI_DOUBLE, right, 2, com, &(rreq[recv]));
			recv = recv + 1;
                        if(ret != MPI_SUCCESS)
                       printf("MPI\n");
		}
		

		if(left >= 0)
		{
			ret = MPI_Irecv(rbuf_lefth, n1, MPI_DOUBLE, left, 1, com, &(rreq[recv]));
			recv = recv + 1;
                        if(ret != MPI_SUCCESS)
                  printf("MPI\n");
		}
		
		if(down >= 0)
		{
			ret = MPI_Irecv(rbuf_downh, n0, MPI_DOUBLE, down, 4, com, &(rreq[recv]));
			recv = recv + 1;
                        if(ret != MPI_SUCCESS)
                  printf("MPI\n");
		}
		
		if(up >= 0)
		{
			ret = MPI_Irecv(rbuf_uph, n0, MPI_DOUBLE, up, 3, com, &(rreq[recv]));
			recv = recv + 1;
                        if(ret != MPI_SUCCESS)
                  printf("MPI\n");
		}
		ret = MPI_Waitall(recv, rreq, MPI_STATUS_IGNORE);
	}
	
	double *prom = NULL;
	
	cudaMalloc(&prom, n00*n11*sizeof(double));
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error malloc\n");
	   fle = true;
	}
	fle = fle || cuda_zero(dev, cudaStreams, prom, n00, n11);

	cuda_pint(dev, cudaStreams, rhs, n0, n1, prom, n00, o2, o3);
	if(nei != 0)
	{
		if(right >= 0)
		{
			cudaMemcpy(rbuf_right, rbuf_righth, n1*sizeof(double), cudaMemcpyHostToDevice);
			fle = fle || cuda_recvm(dev, cudaStreams, prom, n1, rbuf_right, n00, o3, n00-1);
		}
		
		if(left >= 0)
		{
			cudaMemcpy(rbuf_left, rbuf_lefth, n1*sizeof(double), cudaMemcpyHostToDevice);
			fle = fle || cuda_recvm(dev, cudaStreams, prom, n1, rbuf_left, n00, o3, 0);
		}
		
		if(down >= 0)
		{
			cudaMemcpy(rbuf_down, rbuf_downh, n0*sizeof(double), cudaMemcpyHostToDevice);
			fle = fle || cuda_recvm(dev, cudaStreams, prom, n0, rbuf_down, 1, 0, o2);
		}
		
		if(up >= 0)
		{
			cudaMemcpy(rbuf_up, rbuf_uph, n0*sizeof(double), cudaMemcpyHostToDevice);
			fle = cuda_recvm(dev, cudaStreams, prom, n0, rbuf_up, 1, 0, o2+n00*(n11-1));
		}
	}		
			
	fle = fle || cuda_left(dev, cudaStreams, res, prom, n0, n1, o1, o2, o3, o4, n00, hx, hy);		
	cudaFree(prom);
	
	if(nei != 0)
	{
		ret = MPI_Waitall(send, sreq, MPI_STATUS_IGNORE);
		free(sreq);
		free(rreq);
		free(sbuf_righth);
		free(sbuf_lefth);
		free(sbuf_uph);
		free(sbuf_downh);
		free(rbuf_righth);
		free(rbuf_lefth);
		free(rbuf_uph);
		free(rbuf_downh);
	}
	return fle;
}


int main(int argc, char **argv)
{
	int N0, N1;                     // Mesh has N0 x N1 nodes.
    int ProcNum, rank;              // the number of processes and rank in communicator.
    int power, p0, p1;              // ProcNum = 2^(power), power splits into sum p0 + p1.
    int dims[2];                    // dims[0] = 2^p0, dims[1] = 2^p1 (--> M = dims[0]*dims[1]).
    int n0,n1, k0,k1;               // N0 = n0*dims[0] + k0, N1 = n1*dims[1] + k1.
    int Coords[2];                  // the process coordinates in the cartesian topology created for mesh.
    
	const double A = 3.0;
    const double B =  3.0;
	
    MPI_Comm Grid_Comm;             // this is a handler of a new communicator.
    const int ndims = 2;            // the number of a process topology dimensions.
    int periods[2] = {0,0};         // it is used for creating processes topology.
    int left, right, up, down;      // the neighbours of the process.

	double * SolVect = NULL;					    // the solution array
    double * ResVect = NULL;					    // the residual array
	double * PromVect = NULL;
	double * BasicVect = NULL;					    // the vector of A-orthogonal system in CGM
	double * RHS_Vect = NULL;					    // the right hand side of Puasson equation.
	double* x = NULL;
	double* y = NULL;
	double hx, hy;  
	double sp, alpha, tau;	// auxiliary values
	double start, finish;		
	
	MPI_Status status;
	int i,j;
	int o1 = 1;
	int o2 = 1;
	int o3 = 1;
	int o4 = 1;
	double *sbuf_up = NULL;
	double *sbuf_down = NULL;
	double *sbuf_right = NULL;
	double *sbuf_left = NULL;
	
	double *rbuf_up = NULL;
	double *rbuf_down = NULL;
	double *rbuf_right = NULL;
	double *rbuf_left = NULL;

	bool fle = false;
	
    MPI_Init(&argc,&argv);
	start = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD,&ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(argc != 2)
    {
        if(rank == 0)
               printf("Wrong parameter set.\n"
                       "Usage: DomainDecomp <Nodes number_1> <Nodes number_2>\nFinishing...\n");
        MPI_Finalize();
        return(1);
    }
    N0 = atoi(argv[1]); N1 = N0;
    
    if((N0 <= 0)||(N1 <= 0))
    {
        if(rank == 0)
           printf("The first and the second arguments (mesh numbers) should be positive.\n");
        
        MPI_Finalize();
        return(2);
    }
    
    if((power = IsPower(ProcNum)) < 0)
    {
        if(rank == 0)
           printf("The number of procs must be a power of 2.\n");
        MPI_Finalize();
        return(3);
    }
    
    p0 = SplitFunction(N0, N1, power);
    p1 = power - p0;
    
    dims[0] = (unsigned int) 1 << p0;   dims[1] = (unsigned int) 1 << p1;
    n0 = N0 >> p0;                      n1 = N1 >> p1;
    k0 = N0 - dims[0]*n0;               k1 = N1 - dims[1]*n1;

    hx = A / (N0-1);    hy = B / (N1-1);
#ifdef Print
    if(rank == 0)
    {
       printf("The number of processes ProcNum = 2^%d. It is split into %d x %d processes.\n"
               "The number of nodes N0 = %d, N1 = %d. Blocks B(i,j) have size:\n", power, dims[0],dims[1], N0,N1);

	if((k0 > 0)&&(k1 > 0))
	   printf("-->\t %d x %d iff i = 0 .. %d, j = 0 .. %d;\n", n0+1,n1+1, k0-1,k1-1);
        if(k1 > 0)
           printf("-->\t %d x %d iff i = %d .. %d, j = 0 .. %d;\n", n0,n1+1, k0,dims[0]-1, k1-1);
        if(k0 > 0)
           printf("-->\t %d x %d iff i = 0 .. %d, j = %d .. %d;\n", n0+1,n1, k0-1, k1,dims[1]-1);

       printf("-->\t %d x %d iff i = %d .. %d, j = %d .. %d.\n", n0,n1, k0,dims[0]-1, k1,dims[1]-1);
    }
#endif

    // the cartesian topology of processes is being created ...
    int retc = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, TRUE, &Grid_Comm);
	if( retc != MPI_SUCCESS)
	{
		printf("CART\n");
	}
    MPI_Comm_rank(Grid_Comm, &rank);
	MPI_Comm_size(Grid_Comm,&ProcNum);
    MPI_Cart_coords(Grid_Comm, rank, ndims, Coords);
    
    if(Coords[0] < k0)
        ++n0;
    if(Coords[1] < k1)
        ++n1;
    
    MPI_Cart_shift(Grid_Comm, 0, 1, &left, &right);
    MPI_Cart_shift(Grid_Comm, 1, 1, &down, &up);
	
	cudaDeviceProp dev;
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	int devnum = rank%2;
	cudaSetDevice(devnum);
	
	cudaGetDeviceProperties(&dev, devnum);
	
	if(dev.canMapHostMemory != 1 && dev.unifiedAddressing != 1 && dev.major < 2)
	{
		printf("Bad device\n");
		return(1);
	}
	
	cudaMalloc(&SolVect, n0*n1*sizeof(double));
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error malloc\n");
	   fle = true;
	}
	cudaMalloc(&PromVect, n0*n1*sizeof(double));
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error malloc\n");
	   fle = true;
	}
	cudaMalloc(&ResVect, n0*n1*sizeof(double));
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error malloc\n");
	   fle = true;
	}
	cudaMalloc(&RHS_Vect, n0*n1*sizeof(double));
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error malloc\n");
	   fle = true;
	}
	cudaMalloc(&x, n0*sizeof(double));
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error malloc\n");
	   fle = true;
	}
	cudaMalloc(&y, n1*sizeof(double));
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error malloc\n");
	   fle = true;
	}

	cudaStream_t cudaStreams[1];
	
	for(int i = 0; i < 1; i++)
		cudaStreamCreate(cudaStreams + i);

	fle = fle || cuda_setka(dev, cudaStreams[0], x, n0, Coords[0], k0, hx);
	fle = fle || cuda_setka(dev, cudaStreams[0], y, n1, Coords[1], k1, hy);

	fle = fle || cuda_zero(dev, cudaStreams, ResVect, n0, n1);
	
	fle = fle || cuda_zero(dev, cudaStreams, RHS_Vect, n0, n1);
   
	fle = fle || cuda_rpart(dev, cudaStreams, RHS_Vect, n0, n1, x, y);
	
	fle = fle || cuda_zero(dev, cudaStreams, SolVect, n0, n1);
	
	fle = fle || cuda_zero(dev, cudaStreams, PromVect, n0, n1);
	
	int nei = 4;
	
    if(right < 0)
	{
		fle = fle || cuda_gran(dev, cudaStreams[0], SolVect, n1, A, y, n0, n0-1);
		nei = nei - 1;
		o1 = 0;
	}
	
	if(left < 0)
	{
		fle = fle || cuda_gran(dev, cudaStreams[0], SolVect, n1, 0.0, y, n0, 0);
		nei = nei - 1;
		o2 = 0;
	}
	
	if(down < 0)
	{
		fle = fle || cuda_gran(dev, cudaStreams[0], SolVect, n0, 0.0, x, 1, 0);
		nei = nei - 1;
		o3 = 0;
	}
	
	if(up < 0)
	{
		fle = fle || cuda_gran(dev, cudaStreams[0], SolVect, n0, B, x, 1, n0*(n1-1));
		nei = nei - 1;
		o4 = 0;
	}
	
    char str[150];
   
	if((nei != 0)&&(fle == false))
	{
		cudaMalloc(&sbuf_up, n0*sizeof(double));
		if(cudaPeekAtLastError() != cudaSuccess)
	    {
	      printf("error malloc\n");
	      fle = true;
	    }
		cudaMalloc(&sbuf_down, n0*sizeof(double));
		if(cudaPeekAtLastError() != cudaSuccess)
	    {
	      printf("error malloc\n");
	      fle = true;
	    }
		cudaMalloc(&sbuf_right, n1*sizeof(double));
		if(cudaPeekAtLastError() != cudaSuccess)
	    {
	      printf("error malloc\n");
	      fle = true;
	    }
		cudaMalloc(&sbuf_left, n1*sizeof(double));
		if(cudaPeekAtLastError() != cudaSuccess)
	    {
	      printf("error malloc\n");
	      fle = true;
	    }
		cudaMalloc(&rbuf_up, n0*sizeof(double));
		if(cudaPeekAtLastError() != cudaSuccess)
	    {
	      printf("error malloc\n");
	      fle = true;
	    }
		cudaMalloc(&rbuf_down, n0*sizeof(double));
		if(cudaPeekAtLastError() != cudaSuccess)
	    {
	      printf("error malloc\n");
	      fle = true;
	    }
		cudaMalloc(&rbuf_right, n1*sizeof(double));
		if(cudaPeekAtLastError() != cudaSuccess)
	    {
	      printf("error malloc\n");
	      fle = true;
	    }
		cudaMalloc(&rbuf_left, n1*sizeof(double));
		if(cudaPeekAtLastError() != cudaSuccess)
	    {
	      printf("error malloc\n");
	      fle = true;
	    }
	}

    if(fle == false)
	{
		fle = fle || LeftPart(sbuf_up, sbuf_down, sbuf_right, sbuf_left, rbuf_up, rbuf_down, rbuf_right, rbuf_left, dev, cudaStreams, SolVect, ResVect, n0, n1, hx, hy, left, right, down, up, Grid_Comm, nei);
		
		fle = fle || cuda_prisv(dev, cudaStreams, ResVect, RHS_Vect, n0, n1, o1, o2, o3, o4, 1.0, 1.0);
	  
		tau = cuda_scal(dev, cudaStreams, ResVect, ResVect, n0, n1, hx, hy, Grid_Comm, &fle);
	   
		fle = fle || LeftPart(sbuf_up, sbuf_down, sbuf_right, sbuf_left, rbuf_up, rbuf_down, rbuf_right, rbuf_left, dev, cudaStreams, ResVect, PromVect, n0, n1, hx, hy, left, right, down, up, Grid_Comm, nei);
		sp = cuda_scal(dev, cudaStreams, PromVect, ResVect, n0, n1, hx, hy, Grid_Comm, &fle);
		tau = tau/sp;
	  
		fle = fle || cuda_prisv(dev, cudaStreams, SolVect, ResVect, n0, n1, o1, o2, o3, o4, 1.0, tau);
		
		BasicVect = ResVect;    
		cudaMalloc(&ResVect, n0*n1*sizeof(double));
		
		fle = fle || cuda_zero(dev, cudaStreams, ResVect, n0, n1);
		bool fl = false;
		double alarg;
		while((fl == false)&&(fle == false))
		{
			fle = fle || LeftPart(sbuf_up, sbuf_down, sbuf_right, sbuf_left, rbuf_up, rbuf_down, rbuf_right, rbuf_left, dev, cudaStreams, SolVect, ResVect, n0, n1, hx, hy, left, right, down, up, Grid_Comm, nei);
			
			fle = fle || cuda_prisv(dev, cudaStreams, ResVect, RHS_Vect, n0, n1, o1, o2, o3, o4, 1.0, 1.0);
			
			fle = fle || LeftPart(sbuf_up, sbuf_down, sbuf_right, sbuf_left, rbuf_up, rbuf_down, rbuf_right, rbuf_left, dev, cudaStreams, ResVect, PromVect, n0, n1, hx, hy, left, right, down, up, Grid_Comm, nei);
			
			alpha = cuda_scal(dev, cudaStreams, PromVect, BasicVect, n0, n1, hx, hy, Grid_Comm, &fle);
			
			alpha = alpha/sp;
			
			alarg = -1.0*alpha;
			fle = fle || cuda_prisv(dev, cudaStreams, BasicVect, ResVect, n0, n1, o1, o2, o3, o4, alarg, -1.0);
		
			tau = cuda_scal(dev, cudaStreams, ResVect, BasicVect, n0, n1, hx, hy, Grid_Comm, &fle);
			
		
			fle = fle || LeftPart(sbuf_up, sbuf_down, sbuf_right, sbuf_left, rbuf_up, rbuf_down, rbuf_right, rbuf_left, dev, cudaStreams, BasicVect, PromVect, n0, n1, hx, hy, left, right, down, up, Grid_Comm, nei);
			sp = cuda_scal(dev, cudaStreams, PromVect, BasicVect, n0, n1, hx, hy, Grid_Comm, &fle);
			tau = tau/sp;
		   
			fle = fle || cuda_prisv(dev, cudaStreams, PromVect, SolVect, n0, n1, 1, 1, 1, 1, 0.0, -1.0);	
			
			fle = fle || cuda_prisv(dev, cudaStreams, SolVect, BasicVect, n0, n1, o1, o2, o3, o4, 1.0, tau);	
			fl = cuda_crit(dev, cudaStreams, PromVect, SolVect, n0, n1, hx, hy, Grid_Comm, &fle);
		}
		if(fle == false)
		{
			double* Solh = NULL;
			double* xh = NULL;
			double* yh = NULL;
			Solh = (double*)malloc(n0*n1*sizeof(double));
			xh = (double*)malloc(n0*sizeof(double));
			yh = (double*)malloc(n1*sizeof(double));
			cudaMemcpy(Solh, SolVect, n0*n1*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(xh, x, n0*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(yh, y, n1*sizeof(double), cudaMemcpyDeviceToHost);
			
			if (rank == 0)
			{
				FILE *fp;
				sprintf(str,"res_%d_%d.csv", power, N0);
				fp = fopen(str,"w");
				for (j=0; j < n1; j++)
				{
					for (i=0; i < n0; i++)
						fprintf(fp,"%f;%f;%f\n", xh[i], yh[j], Solh[n0*j+i]);
				}
				int size[2];
				double *xr = NULL;
				double *yr = NULL;
				double *solr = NULL;
				int ii;
				for(ii = 1; ii <= ProcNum - 1; ii++)
				{
					MPI_Recv(size, 2, MPI_INT, ii, 0, Grid_Comm, &status);
					xr = (double*)malloc(size[0]*sizeof(double));
					yr = (double*)malloc(size[1]*sizeof(double));
					solr = (double*)malloc(size[0]*size[1]*sizeof(double));
					MPI_Recv(xr, size[0], MPI_DOUBLE, ii, 1, Grid_Comm, &status);
					MPI_Recv(yr, size[1], MPI_DOUBLE, ii, 2, Grid_Comm, &status);
					MPI_Recv(solr, size[0]*size[1], MPI_DOUBLE, ii, 3, Grid_Comm, &status);
					for (j=0; j < size[1]; j++)
					{
					   for (i=0; i < size[0]; i++)
						   fprintf(fp,"%f;%f;%f\n", xr[i], yr[j], solr[size[0]*j+i]);
					}
					free(xr);
					free(yr);
					free(solr);
				}
				finish = MPI_Wtime();
				fprintf(fp,"runtime = %f", finish - start);
				fclose(fp);
			}
			else
			{
				int size[2];
				size[0] = n0;
				size[1] = n1;
				MPI_Send(size, 2, MPI_INT, 0, 0, Grid_Comm);
				MPI_Send(xh, n0, MPI_DOUBLE, 0, 1, Grid_Comm);
				MPI_Send(yh, n1, MPI_DOUBLE, 0, 2, Grid_Comm);
				MPI_Send(Solh, n0*n1, MPI_DOUBLE, 0, 3, Grid_Comm);
			}
			free(Solh); free(xh); free(yh);
        }
		if(nei != 0)
		{
			cudaFree(sbuf_right);
			cudaFree(sbuf_left);
			cudaFree(sbuf_up);
			cudaFree(sbuf_down);
			cudaFree(rbuf_right);
			cudaFree(rbuf_left);
			cudaFree(rbuf_up);
			cudaFree(rbuf_down);
		}
	}
	cudaFree(SolVect); cudaFree(ResVect); cudaFree(BasicVect); cudaFree(RHS_Vect); cudaFree(PromVect); cudaFree(x); cudaFree(y); 
	
	for(int i = 0; i < 1; i++)
		cudaStreamDestroy(cudaStreams[i]);
	
    MPI_Finalize();
    
    return 0;
}
