#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>


#define Print
#define TRUE  ((int) 1)
#define FALSE ((int) 0)

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

double BoundaryValue(double x, double y)
{
   double res;
   res = log(1 + x*y);
   return res;
}

int RightPart(double * rhs, int n0, int n1, double* x, double* y)
{
    int i, j;
    memset(rhs,0,n0*n1*sizeof(double));
    for(j=0; j<n1; j++)
            for(i=0; i<n0; i++)
                rhs[j*n0+i] = (x[i]*x[i] + y[j]*y[j])/((1 + x[i]*y[j])*(1 + x[i]*y[j]));
    return 0;
}

int LeftPart(double * rhs, double * res, int n0, int n1, double hx, double hy, int left, int right, int down, int up, MPI_Comm com, int nei)
{
	double *sbuf_up;
	double *sbuf_down;
	double *sbuf_right;
	double *sbuf_left;
	
	double *rbuf_up;
	double *rbuf_down;
	double *rbuf_right;
	double *rbuf_left;
	int send = 0;
	int recv = 0;
	int ret;
	int o1 = 0;
	int o2 = 0;
	int o3 = 0;
	int o4 = 0;
	int n00;
	int n11;
    int i,j;
	MPI_Request *sreq;
	MPI_Request *rreq;
	//printf("1\n");
    n00 = n0;
    n11 = n1;	
	
	if (nei != 0)
	{
		sbuf_up = (double *)malloc(n0*sizeof(double));
		sbuf_down = (double *)malloc(n0*sizeof(double));
		sbuf_right = (double *)malloc(n1*sizeof(double));
		sbuf_left = (double *)malloc(n1*sizeof(double));
		rbuf_up = (double *)malloc(n0*sizeof(double));
		rbuf_down = (double *)malloc(n0*sizeof(double));
		rbuf_right = (double *)malloc(n1*sizeof(double));
		rbuf_left = (double *)malloc(n1*sizeof(double));
		sreq = (MPI_Request *)malloc(nei*sizeof(MPI_Request));
		rreq = (MPI_Request *)malloc(nei*sizeof(MPI_Request));
               //printf("1.1.1\n");
		if(right >= 0)
		{
			#pragma omp parallel
		    #pragma omp for schedule (static)
			for(j=0; j<n1; j++)
			{
			  sbuf_right[j] = rhs[n0*j+(n0-1)];
			}
			ret = MPI_Isend(sbuf_right, n1, MPI_DOUBLE, right, 1, com, &(sreq[send]));
			send = send + 1;
			n00 = n00 + 1;
			o1 = 1;
		}
		
		if(left >= 0)
		{
			#pragma omp parallel
		    #pragma omp for schedule (static)
			for(j=0; j<n1; j++)
			{
			  sbuf_left[j] = rhs[n0*j];
			}
			ret = MPI_Isend(sbuf_left, n1, MPI_DOUBLE, left, 2, com, &(sreq[send]));
			send = send + 1;
			n00 = n00 + 1;
			o2 = 1;
		}
		
		if(down >= 0)
		{
			#pragma omp parallel
		    #pragma omp for schedule (static)
			for(i=0; i<n0; i++)
			{
			  sbuf_down[i] = rhs[i];
			}
			ret = MPI_Isend(sbuf_down, n0, MPI_DOUBLE, down, 3, com, &(sreq[send]));
			send = send + 1;
			n11 = n11 + 1;
			o3 = 1;
		}
		
		if(up >= 0)
		{
			#pragma omp parallel
		    #pragma omp for schedule (static)
			for(i=0; i<n0; i++)
			{
			  sbuf_up[i] = rhs[n0*(n1-1)+i];
			}
			ret = MPI_Isend(sbuf_up, n0, MPI_DOUBLE, up, 4, com, &(sreq[send]));
			send = send + 1;
			n11 = n11 + 1;
			o4 = 1;
		}
		
                //ret = MPI_Waitall(send, sreq, MPI_STATUS_IGNORE);  
               //printf("send = %d\n", send);
		if(right >= 0)
		{
			ret = MPI_Irecv(rbuf_right, n1, MPI_DOUBLE, right, 2, com, &(rreq[recv]));
			recv = recv + 1;
                        if(ret != MPI_SUCCESS)
                       printf("MPI\n");
		}
		

		if(left >= 0)
		{
			ret = MPI_Irecv(rbuf_left, n1, MPI_DOUBLE, left, 1, com, &(rreq[recv]));
			recv = recv + 1;
                        if(ret != MPI_SUCCESS)
                  printf("MPI\n");
		}
		
		if(down >= 0)
		{
			ret = MPI_Irecv(rbuf_down, n0, MPI_DOUBLE, down, 4, com, &(rreq[recv]));
			recv = recv + 1;
                        if(ret != MPI_SUCCESS)
                  printf("MPI\n");
		}
		
		if(up >= 0)
		{
			ret = MPI_Irecv(rbuf_up, n0, MPI_DOUBLE, up, 3, com, &(rreq[recv]));
			recv = recv + 1;
                        if(ret != MPI_SUCCESS)
                  printf("MPI\n");
		}
		//printf("recv = %d\n",recv);
		ret = MPI_Waitall(recv, rreq, MPI_STATUS_IGNORE);
               //printf("1.1\n");
	}
	
	double *prom;
	
	prom = (double*)malloc(n00*n11*sizeof(double));
	memset(prom,0,n00*n11*sizeof(double));
	
	#pragma omp parallel
	#pragma omp for schedule (static)
	for(j=0; j < n1; j++)
			for(i=0; i < n0; i++)
	            prom[n00*(j +o3)+i + o2] = rhs[n0*j+i];

	if(nei != 0)
	{
		if(right >= 0)
		{
			#pragma omp parallel
		    #pragma omp for schedule (static)
			for(j=0; j<n1; j++)
			{
			  prom[n00*(j + o3)+(n00-1)] = rbuf_right[j];
			}
		}
		
		if(left >= 0)
		{
			#pragma omp parallel
		    #pragma omp for schedule (static)
			for(j=0; j<n1; j++)
			{
			  prom[n00*(j+o3)] = rbuf_left[j];
			}
		}
		
		if(down >= 0)
		{
			#pragma omp parallel
		    #pragma omp for schedule (static)
			for(i=0; i<n0; i++)
			{
			  prom[i + o2] = rbuf_down[i];
			}
		}
		
		if(up >= 0)
		{
			#pragma omp parallel
		    #pragma omp for schedule (static)
			for(i=0; i<n0; i++)
			{
			  prom[n00*(n11-1)+i + o2] = rbuf_up[i];
			}
		}
	}		
	#pragma omp parallel
	#pragma omp for schedule (static) private(i)
	for(j=1-o3; j < n1-1+o4; j++)
			for(i=1-o2; i < n0-1+o1; i++)
				res[n0*j+i] = (-(prom[n00*(j+o3)+i+o2+1]-prom[n00*(j+o3)+i+o2])/hx+(prom[n00*(j+o3)+i+o2]-prom[n00*(j+o3)+i+o2-1])/hx)/hx+(-(prom[n00*(j+1+o3)+i+o2]-prom[n00*(j+o3)+i+o2])/hy+(prom[n00*(j+o3)+i+o2]-prom[n00*(j-1+o3)+i+o2])/hy)/hy;
	free(prom);
	
	if(nei != 0)
	{
		ret = MPI_Waitall(send, sreq, MPI_STATUS_IGNORE);
		free(sreq);
		free(rreq);
		free(sbuf_right);
		free(sbuf_left);
		free(sbuf_up);
		free(sbuf_down);
		free(rbuf_right);
		free(rbuf_left);
		free(rbuf_up);
		free(rbuf_down);
	}
	//printf("2\n");
	return 0;
}

double scal(double * v1, double * v2, int n0, int n1, double hx, double hy, MPI_Comm com)
{
	double sc = 0;
	int i,j;
	#pragma omp parallel
	#pragma omp for schedule (static) reduction(+:sc)
	for(j=0; j < n1; j++)
			for(i=0; i < n0; i++)
				sc = sc + v1[n0*j+i]*v2[n0*j+i]*hx*hy;
			
	double glob_sc = 0;
	
	int ret = MPI_Allreduce(&sc, &glob_sc, 1, MPI_DOUBLE, MPI_SUM, com);
	
	return glob_sc;
}

int crit(double * v1, double * v2, int n0, int n1, MPI_Comm com)
{
	double norm = 0;
	double parnorm = 0;
	double eps = 0.0001;
	double glob_norm = 0;
	int i,j;
	
	#pragma omp parallel firstprivate (norm)
	{
		#pragma omp for schedule (static)
	    for(j=0; j < n1; j++)
		     for(i=0; i < n0; i++)
                         if(norm < fabs(v1[n0*j+i]-v2[n0*j+i]))
		         	 norm = fabs(v1[n0*j+i]-v2[n0*j+i]);
		#pragma omp critical
		{
			if(parnorm < norm)
		         	 parnorm = norm;
		}
	
	}
	int ret = MPI_Allreduce(&parnorm, &glob_norm, 1, MPI_DOUBLE, MPI_MAX, com);
 
        //printf("norm = %f\n",glob_norm);
	
	if(glob_norm < eps)
	{
		return 1;
	}
	else
	{
		return 0;
	}
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

	double * SolVect;					    // the solution array
    double * ResVect;					    // the residual array
	double * PromVect;
	double * BasicVect;					    // the vector of A-orthogonal system in CGM
	double * RHS_Vect;					    // the right hand side of Puasson equation.
	double* x;
	double* y;
	double hx, hy;  
	double sp, alpha, tau, NewValue, err;	// auxiliary values
	double start, finish;
	int SDINum, CGMNum;					    // the number of steep descent and CGM iterations.
	int counter;		
	
	MPI_Status status;
	int i,j;
	int o1 = 1;
	int o2 = 1;
	int o3 = 1;
	int o4 = 1;
	int required = MPI_THREAD_FUNNELED, provided;
    // MPI Library is being activated ...
    MPI_Init_thread(&argc,&argv, required, &provided);
	start = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD,&ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(argc != 3)
    {
        if(rank == 0)
               printf("Wrong parameter set.\n"
                       "Usage: DomainDecomp <Nodes number_1> <Nodes number_2>\nFinishing...\n");
        MPI_Finalize();
        return(1);
    }
    N0 = atoi(argv[1]); N1 = atoi(argv[2]);
    
    if((N0 <= 0)||(N1 <= 0))
    {
        if(rank == 0)
           //printf("The first and the second arguments (mesh numbers) should be positive.\n");
        
        MPI_Finalize();
        return(2);
    }
    
    if((power = IsPower(ProcNum)) < 0)
    {
        if(rank == 0)
           //printf("The number of procs must be a power of 2.\n");
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
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, TRUE, &Grid_Comm);
    MPI_Comm_rank(Grid_Comm, &rank);
	MPI_Comm_size(Grid_Comm,&ProcNum);
    MPI_Cart_coords(Grid_Comm, rank, ndims, Coords);
    
    if(Coords[0] < k0)
        ++n0;
    if(Coords[1] < k1)
        ++n1;
    
    MPI_Cart_shift(Grid_Comm, 0, 1, &left, &right);
    MPI_Cart_shift(Grid_Comm, 1, 1, &down, &up);
    //printf("(%d,%d) %d %d %d %d", Coords[0],Coords[1], right, left, up, down);
	
	SolVect   = (double *)malloc(n0*n1*sizeof(double));
	PromVect   = (double *)malloc(n0*n1*sizeof(double));
	ResVect   = (double *)malloc(n0*n1*sizeof(double));
	RHS_Vect  = (double *)malloc(n0*n1*sizeof(double));
    x = (double *)malloc(n0*sizeof(double));
	y = (double *)malloc(n1*sizeof(double));
	
	//omp_set_num_threads(3);
	
	//#pragma omp parallel
	//printf("OMP\n");
	
	    #pragma omp parallel
		#pragma omp for schedule (static)
    for(i=0; i<n0; i++)
              if(Coords[0] >= k0)
                x[i] = k0*(n0+1)*hx + (Coords[0]-k0)*n0*hx + i*hx;
              else
		x[i] = Coords[0]*n0*hx + i*hx;
	#pragma omp parallel
	   #pragma omp for schedule (static)
	for(i=0; i<n1; i++)
		if(Coords[1] >= k1)
                y[i] = k1*(n1+1)*hy + (Coords[1]-k1)*n1*hy + i*hy;
              else
		y[i] = Coords[1]*n1*hy + i*hy;
	memset(ResVect,0,n0*n1*sizeof(double));
    RightPart(RHS_Vect,n0,n1,x,y);
	memset(SolVect,0,n0*n1*sizeof(double));
	memset(PromVect,0,n0*n1*sizeof(double));
	
	int nei = 4;
	//printf("0\n");
    if(right < 0)
	{
		#pragma omp parallel
		#pragma omp for schedule (static)
		for(j=0; j<n1; j++)
	    {
		  SolVect[n0*j+(n0-1)] = BoundaryValue(A,y[j]);
	    }
		nei = nei - 1;
		o1 = 0;
	}
	
	if(left < 0)
	{
		#pragma omp parallel
		#pragma omp for schedule (static)
		for(j=0; j<n1; j++)
	    {
		  SolVect[n0*j] = BoundaryValue(0.0,y[j]);
	    }
		nei = nei - 1;
		o2 = 0;
	}
	
	if(down < 0)
	{
		#pragma omp parallel
		#pragma omp for schedule (static)
		for(i=0; i<n0; i++)
	    {
		  SolVect[i] = BoundaryValue(x[i],0.0);
	    }
		nei = nei - 1;
		o3 = 0;
	}
	
	if(up < 0)
	{
		#pragma omp parallel
		#pragma omp for schedule (static)
		for(i=0; i<n0; i++)
	    {
		  SolVect[n0*(n1-1)+i] = BoundaryValue(x[i],B);
	    }
		nei = nei - 1;
		o4 = 0;
	}
        char str[150];
        /*sprintf(str,"res_%d_%d.txt", Coords[0], Coords[1]);
        FILE *fp = fopen(str,"w");
		fprintf(fp,"# This is the conjugate gradient method for descrete Puasson equation.\n"
				"# A = %f, B = %f, N[0,A] = %d, N[0,B] = %d.\n",\
				A, B, n0, n1);
		for (j=0; j < n1; j++)
		{
			for (i=0; i < n0; i++)
				fprintf(fp,"\n%f %f %f", x[i], y[j], SolVect[n0*j+i]);
			fprintf(fp,"\n");
		}
	fclose(fp);*/
	//printf("0.1\n");
	for(counter=1; counter<=1; counter++)
	{
       //printf("0.2\n");
// The residual vector r(k) = Ax(k)-f is calculating ...
        LeftPart(SolVect, ResVect, n0, n1, hx, hy, left, right, down, up, Grid_Comm, nei);
		#pragma omp parallel
		#pragma omp for schedule (static)
		for(j=1-o3; j < n1-1+o4; j++)
			for(i=1-o2; i < n0-1+o1; i++)
				ResVect[n0*j+i] = ResVect[n0*j+i] - RHS_Vect[n0*j+i];
      //printf("0.3\n");
// The value of product (r(k),r(k)) is calculating ...
		tau = scal(ResVect, ResVect, n0, n1, hx, hy, Grid_Comm);
       //printf("0.4\n");
// The value of product sp = (Ar(k),r(k)) is calculating ...
        LeftPart(ResVect, PromVect, n0, n1, hx, hy, left, right, down, up, Grid_Comm, nei);
		sp = scal(PromVect, ResVect, n0, n1, hx, hy, Grid_Comm);
		tau = tau/sp;
      //printf("0.5\n");
// The x(k+1) is calculating ...
        #pragma omp parallel
		#pragma omp for schedule (static)
		for(j=1-o3; j < n1-1+o4; j++)
			for(i=1-o2; i < n0-1+o1; i++)
			{
				NewValue = SolVect[n0*j+i]-tau*ResVect[n0*j+i];
				SolVect[n0*j+i] = NewValue;
			}
    }
	//printf("0.6\n");
	BasicVect = ResVect;    // g(0) = r(k-1).
    ResVect = (double *)malloc(n0*n1*sizeof(double));
    memset(ResVect,0,n0*n1*sizeof(double));
	int fl = 0;
	
	while(fl == 0)
	{
		// The residual vector r(k) is calculating ...
		LeftPart(SolVect, ResVect, n0, n1, hx, hy, left, right, down, up, Grid_Comm, nei);
		#pragma omp parallel
		#pragma omp for schedule (static)
		for(j=1-o3; j < n1-1+o4; j++)
			for(i=1-o2; i < n0-1+o1; i++)
				ResVect[n0*j+i] = ResVect[n0*j+i] - RHS_Vect[n0*j+i];

	// The value of product (Ar(k),g(k-1)) is calculating ...
	    memset(PromVect,0,n0*n1*sizeof(double));
		LeftPart(ResVect, PromVect, n0, n1, hx, hy, left, right, down, up, Grid_Comm, nei);
		alpha = scal(PromVect, BasicVect, n0, n1, hx, hy, Grid_Comm);
		alpha = alpha/sp;
        memset(PromVect,0,n0*n1*sizeof(double));
           //printf("alpha = %f\n", alpha);
	// The new basis vector g(k) is being calculated ...
	    #pragma omp parallel
		#pragma omp for schedule (static)
		for(j=1-o3; j < n1-1+o4; j++)
			for(i=1-o2; i < n0-1+o1; i++)
				BasicVect[n0*j+i] = ResVect[n0*j+i]-alpha*BasicVect[n0*j+i];

	// The value of product (r(k),g(k)) is being calculated ...
		tau = scal(ResVect, BasicVect, n0, n1, hx, hy, Grid_Comm);
		
	// The value of product sp = (Ag(k),g(k)) is being calculated ...
	    LeftPart(BasicVect, PromVect, n0, n1, hx, hy, left, right, down, up, Grid_Comm, nei);
		sp = scal(PromVect, BasicVect, n0, n1, hx, hy, Grid_Comm);
		tau = tau/sp;
        memset(PromVect,0,n0*n1*sizeof(double));
	// The x(k+1) is being calculated ...
		err = 0.0;
		#pragma omp parallel
		#pragma omp for schedule (static)
		for(j=0; j < n1; j++)
			for(i=0; i < n0; i++)
				PromVect[n0*j+i] = SolVect[n0*j+i];
			
        //printf("tau = %f\n", tau);
		#pragma omp parallel
		#pragma omp for schedule (static)
		for(j=1-o3; j < n1-1+o4; j++)
			for(i=1-o2; i < n0-1+o1; i++)
			{
				SolVect[n0*j+i] = SolVect[n0*j+i]-tau*BasicVect[n0*j+i];
			}
			
		fl = crit(PromVect, SolVect, n0, n1, Grid_Comm);
               // printf("iteration\n");
	}
	
	if (rank == 0)
	{
		FILE *fp;
		sprintf(str,"resp_%d_%d.csv", power, N0);
	    fp = fopen(str,"w");
		//fprintf(fp,"# This is the conjugate gradient method for descrete Puasson equation.\n"
			//	"# A = %f, B = %f, N[0,A] = %d, N[0,B] = %d.\n",\
				//A, B, n0, n1);
		for (j=0; j < n1; j++)
		{
			for (i=0; i < n0; i++)
				fprintf(fp,"%f;%f;%f\n", x[i], y[j], SolVect[n0*j+i]);
		}
		/*int size[2];
		double *xr;
		double *yr;
		double *solr;
		int ii;
		//#pragma omp parallel
		//#pragma omp for schedule (static) private(xr, yr, solr, size, j, i, status)
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
		}*/
		finish = MPI_Wtime();
		fprintf(fp,"runtime = %f", finish - start);
	    fclose(fp);
	}
	else
	{
		int size[2];
		/*size[0] = n0;
		size[1] = n1;
		MPI_Send(size, 2, MPI_INT, 0, 0, Grid_Comm);
		MPI_Send(x, n0, MPI_DOUBLE, 0, 1, Grid_Comm);
		MPI_Send(y, n1, MPI_DOUBLE, 0, 2, Grid_Comm);
		MPI_Send(SolVect, n0*n1, MPI_DOUBLE, 0, 3, Grid_Comm);*/
	}

	free(SolVect); free(ResVect); free(BasicVect); free(RHS_Vect); free(PromVect); free(x); free(y);
	
    MPI_Finalize();
    
    return 0;
}
