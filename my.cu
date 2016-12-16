#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <utility> 
#include <math.h>
#include <mpi.h>
#include <cuda_runtime.h>

using std::min;
using std::max;
using std::swap;

__global__ void cudakernel_setka(double* const x, int n, int Coords, int k, double h)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int i = tid; i < n; i += tnum)
   {
        if(Coords >= k)
                x[i] = k*(n+1)*h + (Coords-k)*n*h + i*h;
        else
		       x[i] = Coords*n*h + i*h;
   }
}
 

bool cuda_setka(cudaDeviceProp dev, cudaStream_t cudaStreams, double* const x, int n, int Coords, int k, double h)
{
    dim3 gridDim;
	dim3 blockDim;
    int task = n;
    int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
    int tpt = (task-1)/maxth + 1;
    int dtn = (task-1)/tpt + 1;
    int avt = 512;
    int dtpb = min(dtn, avt);
    blockDim = dim3(dtpb);
    int dbn = (dtn - 1)/blockDim.x + 1;
    gridDim = dim3(dbn);
    cudakernel_setka<<<gridDim, blockDim, 0, cudaStreams>>>(x, n, Coords, k, h);
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error setka\n");
	   return true;
	}
	cudaStreamSynchronize(cudaStreams);
	return false;
}



__global__ void cudakernel_gran(double* const x, int n, double a, const double* const b, int c1, int c2)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int i = tid; i < n; i += tnum)
   {
        x[c1*i +c2] = log(1 + a*b[i]);
   }
}
 

bool cuda_gran(cudaDeviceProp dev, cudaStream_t cudaStreams, double* const x, int n, double a, const double* const b, int c1, int c2)
{
    dim3 gridDim;
	dim3 blockDim;
    int task = n;
    int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
    int tpt = (task-1)/maxth + 1;
    int dtn = (task-1)/tpt + 1;
    int avt = 512;
    int dtpb = min(dtn, avt);
    blockDim = dim3(dtpb);
    int dbn = (dtn - 1)/blockDim.x + 1;
    gridDim = dim3(dbn);
    cudakernel_gran<<<gridDim, blockDim, 0, cudaStreams>>>(x, n, a, b, c1, c2);
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error gran\n");
	   return true;
	}
	cudaStreamSynchronize(cudaStreams);
	return false;
}


__global__ void cudakernel_rpart(double* const rhs, int n0, int n1, const double* const x, const double* const y)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int k = tid; true; k += tnum)
   {
        int i = k%n0;
		int j = k/n0;
		if (j < n1)
		{
		  rhs[j*n0+i] = (x[i]*x[i] + y[j]*y[j])/((1 + x[i]*y[j])*(1 + x[i]*y[j]));
		}
		else
		{
		   break;
		}
   }
}
 

bool cuda_rpart(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* const rhs, int n0, int n1, const double* const x, const double* const y)
{
    dim3 gridDim;
	dim3 blockDim;
    int task = n0*n1;
    int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
    int tpt = (task-1)/maxth + 1;
    int dtn = (task-1)/tpt + 1;
    int avt = 512;
    int dtpb = min(dtn, avt);
    blockDim = dim3(dtpb);
    int dbn = (dtn - 1)/blockDim.x + 1;
    gridDim = dim3(dbn);
    cudakernel_rpart<<<gridDim, blockDim, 0, cudaStreams[0]>>>(rhs, n0, n1, x, y);
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error rpart\n");
	   return true;
	}
	cudaStreamSynchronize(cudaStreams[0]);
	return false;
}


__global__ void cudakernel_prisv(double* const x, const double* const y, int n0, int n1, int o1, int o2, int o3, int o4, double a, double b)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int k = tid; true; k += tnum)
   {
        int i = 1-o2 + k%(n0-2+o1+o2);
		int j = 1-o3 + k/(n0-2+o1+o2);
		if (j < n1-1+o4)
		{
		  x[j*n0+i] = a*x[j*n0+i] - b*y[j*n0+i];
		}
		else
		{
		   break;
		}
   }
}
 

bool cuda_prisv(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* const x, const double* const y, int n0, int n1, int o1, int o2, int o3, int o4, double a, double b)
{
    dim3 gridDim;
	dim3 blockDim;
    int task = (n0-2+o1+o2)*(n1-2+o3+o4);
    int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
    int tpt = (task-1)/maxth + 1;
    int dtn = (task-1)/tpt + 1;
    int avt = 512;
    int dtpb = min(dtn, avt);
    blockDim = dim3(dtpb);
    int dbn = (dtn - 1)/blockDim.x + 1;
    gridDim = dim3(dbn);
    cudakernel_prisv<<<gridDim, blockDim, 0, cudaStreams[0]>>>(x, y, n0, n1, o1, o2, o3, o4, a, b);
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error prisv\n");
	   return true;
	}
	cudaStreamSynchronize(cudaStreams[0]);
	return false;
}


__global__ void cudakernel_sendm(double* x, int n, const double* b, int c1, int c2)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int i = tid; i < n; i += tnum)
   {
        x[i] = b[c1*i+c2];
   }
}
 

bool cuda_sendm(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* x, int n, const double* b, int c1, int c2)
{
    dim3 gridDim;
	dim3 blockDim;
    int task = n;
    int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
    int tpt = (task-1)/maxth + 1;
    int dtn = (task-1)/tpt + 1;
    int avt = 512;
    int dtpb = min(dtn, avt);
    blockDim = dim3(dtpb);
    int dbn = (dtn - 1)/blockDim.x + 1;
    gridDim = dim3(dbn);
    cudakernel_sendm<<<gridDim, blockDim, 0, cudaStreams[0]>>>(x, n, b, c1, c2);
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error sendm\n");
	   return true;
	}
	cudaStreamSynchronize(cudaStreams[0]);
	return false;
}


__global__ void cudakernel_pint(const double* const rhs, int n0, int n1, double* const x, int n00, int o2, int o3)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int k = tid; true; k += tnum)
   {
        int i = k%n0;
		int j = k/n0;
		if (j < n1)
		{
		  x[n00*(j +o3)+i + o2] = rhs[n0*j+i];
		}
		else
		{
		   break;
		}
   }
}
 

bool cuda_pint(cudaDeviceProp dev, cudaStream_t* cudaStreams, const double* const rhs, int n0, int n1, double* const x, int n00, int o2, int o3)
{
    dim3 gridDim;
	dim3 blockDim;
    int task = n0*n1;
    int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
    int tpt = (task-1)/maxth + 1;
    int dtn = (task-1)/tpt + 1;
    int avt = 512;
    int dtpb = min(dtn, avt);
    blockDim = dim3(dtpb);
    int dbn = (dtn - 1)/blockDim.x + 1;
    gridDim = dim3(dbn);
    cudakernel_pint<<<gridDim, blockDim, 0, cudaStreams[0]>>>(rhs, n0, n1, x, n00, o2, o3);
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error pint\n");
	   return true;
	}
	cudaStreamSynchronize(cudaStreams[0]);
	return false;
}


__global__ void cudakernel_recvm(double* x, int n, const double* b, int c1, int c2, int c3)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int i = tid; i < n; i += tnum)
   {
        x[c1*(i+c2) +c3] = b[i];
   }
}
 

bool cuda_recvm(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* x, int n,const double* b, int c1, int c2, int c3)
{
    dim3 gridDim;
	dim3 blockDim;
    int task = n;
    int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
    int tpt = (task-1)/maxth + 1;
    int dtn = (task-1)/tpt + 1;
    int avt = 512;
    int dtpb = min(dtn, avt);
    blockDim = dim3(dtpb);
    int dbn = (dtn - 1)/blockDim.x + 1;
    gridDim = dim3(dbn);
    cudakernel_recvm<<<gridDim, blockDim, 0, cudaStreams[0]>>>(x, n, b, c1, c2, c3);
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error recvm\n");
	   return true;
	}
	cudaStreamSynchronize(cudaStreams[0]);
	return false;
}


__global__ void cudakernel_left(double* res, const double* const prom, int n0, int n1, int o1, int o2, int o3, int o4, int n00, double hx, double hy)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int k = tid; true; k += tnum)
   {
        int i = 1-o2 + k%(n0-2+o1+o2);
		int j = 1-o3 + k/(n0-2+o1+o2);
		if (j < n1-1+o4)
		{
		  res[n0*j+i] = (-(prom[n00*(j+o3)+i+o2+1]-prom[n00*(j+o3)+i+o2])/hx+(prom[n00*(j+o3)+i+o2]-prom[n00*(j+o3)+i+o2-1])/hx)/hx+(-(prom[n00*(j+1+o3)+i+o2]-prom[n00*(j+o3)+i+o2])/hy+(prom[n00*(j+o3)+i+o2]-prom[n00*(j-1+o3)+i+o2])/hy)/hy;
		}
		else
		{
		   break;
		}
   }
}
 

bool cuda_left(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* x, const double* const y, int n0, int n1, int o1, int o2, int o3, int o4, int n00, double hx, double hy)
{
    dim3 gridDim;
	dim3 blockDim;
    int task = (n0-2+o1+o2)*(n1-2+o3+o4);
    int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
    int tpt = (task-1)/maxth + 1;
    int dtn = (task-1)/tpt + 1;
    int avt = 512;
    int dtpb = min(dtn, avt);
    blockDim = dim3(dtpb);
    int dbn = (dtn - 1)/blockDim.x + 1;
    gridDim = dim3(dbn);
    cudakernel_left<<<gridDim, blockDim, 0, cudaStreams[0]>>>(x, y, n0, n1, o1, o2, o3, o4, n00, hx, hy);
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error left\n");
	   return true;
	}
	cudaStreamSynchronize(cudaStreams[0]);
	return false;
}


__global__ void cudakernel_prscal(double * const sum, const double * const v1, const double * const v2, const int n0, const int n1, const double hx, const double hy)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int k = tid; true; k += tnum)
   {
        int i = k%n0;
		int j = k/n0;
		if (j < n1)
		{
		  sum[k] = v1[j*n0+i]*v2[j*n0+i]*hx*hy;
		}
		else
		{
		   break;
		}
   }
}


__global__ void cudakernel_scal(double* const sum2, const double* const sum1, const int task, const int dbn)
{
   extern __shared__ double data [];
   
   int blid = blockIdx.x;
   if(blid < dbn)
   {
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      int tnum = gridDim.x*blockDim.x;
	  double sum = 0;
	  for(int k = tid; k < task; k += tnum)
	   sum = sum + sum1[k];
	  
	  int tlbid = threadIdx.x;
	  data[tlbid] = sum;
	  __syncthreads ();
	  
	  int bsize = blockDim.x;
	  for(int s = 1; s< bsize; s*= 2)
	  {
	      if((tlbid %(2*s) == 0)and(tlbid + s<bsize))
		  {
		      data[tlbid] += data[tlbid+s];
		  }
		  __syncthreads ();
	  }
	  if(tlbid == 0)
	  {
	     sum2[blid] = data[0];
	  }
   }
}


double cuda_scal(cudaDeviceProp dev, cudaStream_t* cudaStreams, const double * const v1, const double * const v2,const int n0,const int n1,const double hx,const double hy, MPI_Comm com, bool* fle)
{
   dim3 gridDim;
   dim3 blockDim;
   double* sum1;
   double* sum2;
   bool* fl = fle;
   cudaMalloc(&sum1, n0*n1*sizeof(double));
   cudaMalloc(&sum2, n0*n1*sizeof(double));
   int task = n0*n1;
   int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
   int tpt = (task-1)/maxth + 1;
   int dtn = (task-1)/tpt + 1;
   int avt = 512;
   int dtpb = min(dtn, avt);
   blockDim = dim3(dtpb);
   int dbn = (dtn - 1)/blockDim.x + 1;
   gridDim = dim3(dbn);
   cudakernel_prscal<<<gridDim, blockDim, 0, cudaStreams[0]>>>(sum1, v1, v2, n0, n1, hx, hy);
   if(cudaPeekAtLastError() != cudaSuccess)
   {
	   printf("error prscal\n");
	   *fl = true;
   }
   while(*fl == false)
   {
      cudakernel_scal<<<gridDim, blockDim, dtpb*sizeof(double), cudaStreams[0]>>>(sum2, sum1, task, dbn);  
	  if(cudaPeekAtLastError() != cudaSuccess)
	  {
	     printf("error scal\n");
	     *fl = true;
	  }
	  task = dbn;
	  if(task == 1)
	  {
	     break;
	  }
      maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
      tpt = (task-1)/maxth + 1;
      dtn = (task-1)/tpt + 1;
      avt = dev.maxThreadsPerBlock/2;
      dtpb = min(dtn, avt);
      blockDim = dim3(dtpb);
      dbn = (dtn - 1)/blockDim.x + 1;
      gridDim = dim3(dbn);
	  swap(sum1, sum2);
	  //cudakernel_prisv<<<gridDim, blockDim, 0, cudaStreams[0]>>>(sum1, sum2, n0, n1,  1, 1, 1, 1, 0.0, -1.0);
	//cudaStreamSynchronize(cudaStreams[0]);
   }
   double glob_sc = 0;
   double sc = 0;
   if (*fl == false)
      cudaMemcpy(&sc, sum2, sizeof(sc), cudaMemcpyDeviceToHost);
   int ret = MPI_Allreduce(&sc, &glob_sc, 1, MPI_DOUBLE, MPI_SUM, com);
   if(ret != MPI_SUCCESS)
        printf("MPI\n");
   cudaFree(sum1);
   cudaFree(sum2);
   return glob_sc;
}



__global__ void cudakernel_prcrit(double * const sum, const double * const v1, const double * const v2, const int n0, const int n1,const double hx,const double hy)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int k = tid; true; k += tnum)
   {
        int i = k%n0;
		int j = k/n0;
		if (j < n1)
		{
		  sum[k] = fabs(v1[j*n0+i]-v2[j*n0+i]);
		}
		else
		{
		   break;
		}
   }
}


__global__ void cudakernel_crit(double* const sum2, const double* const sum1, const int task, const int dbn)
{
   extern __shared__ double data [];
   
   int blid = blockIdx.x;
   if(blid < dbn)
   {
      int tid = threadIdx.x + blockIdx.x*blockDim.x;
      int tnum = gridDim.x*blockDim.x;
	  double sum = 0;
	  for(int k = tid; k < task; k += tnum)
	   sum = max(sum, sum1[k]);
	  
	  int tlbid = threadIdx.x;
	  data[tlbid] = sum;
	  __syncthreads ();
	  
	  int bsize = blockDim.x;
	  for(int s = 1; s< bsize; s*= 2)
	  {
	      if((tlbid %(2*s) == 0)and(tlbid + s<bsize))
		  {
		      data[tlbid] = max(data[tlbid], data[tlbid+s]);
		  }
		  __syncthreads ();
	  }
	  if(tlbid == 0)
	  {
	     sum2[blid] = data[0];
	  }
   }
}


bool cuda_crit(cudaDeviceProp dev, cudaStream_t* cudaStreams, const double * v1, const double * v2, const int n0, const int n1, const double hx, const double hy, MPI_Comm com, bool* fle)
{
   dim3 gridDim;
   dim3 blockDim;
   double* sum1;
   double* sum2;
   bool* fl = fle;
   cudaMalloc(&sum1, n0*n1*sizeof(double));
   cudaMalloc(&sum2, n0*n1*sizeof(double));
   int task = n0*n1;
   int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
   int tpt = (task-1)/maxth + 1;
   int dtn = (task-1)/tpt + 1;
   int avt = 512;
   int dtpb = min(dtn, avt);
   blockDim = dim3(dtpb);
   int dbn = (dtn - 1)/blockDim.x + 1;
   gridDim = dim3(dbn);
   cudakernel_prcrit<<<gridDim, blockDim, 0, cudaStreams[0]>>>(sum1, v1, v2, n0, n1, hx, hy);
   if(cudaPeekAtLastError() != cudaSuccess)
   {
	   printf("error prcrit\n");
	   *fl = true;
   }
   while(*fl == false)
   {
      cudakernel_crit<<<gridDim, blockDim, dtpb*sizeof(double), cudaStreams[0]>>>(sum2, sum1, task, dbn);  
	  if(cudaPeekAtLastError() != cudaSuccess)
      {
	     printf("error crit\n");
	     *fl = true;
      }
	  task = dbn;
	  if(task == 1)
	  {
	     break;
	  }
      maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
      tpt = (task-1)/maxth + 1;
      dtn = (task-1)/tpt + 1;
      avt = dev.maxThreadsPerBlock/2;
      dtpb = min(dtn, avt);
      blockDim = dim3(dtpb);
      dbn = (dtn - 1)/blockDim.x + 1;
      gridDim = dim3(dbn);
	  swap(sum1, sum2);
   }
   double eprr = 0.0001;
   double glob_norm = 0;
   double norm = 0;
   if(*fl == false)
     cudaMemcpy(&norm, sum2, sizeof(norm), cudaMemcpyDeviceToHost);
   int ret = MPI_Allreduce(&norm, &glob_norm, 1, MPI_DOUBLE, MPI_MAX, com);
   if(ret != MPI_SUCCESS)
        printf("MPI\n");

   cudaFree(sum1);
   cudaFree(sum2);
   
   return glob_norm < eprr;
}


__global__ void cudakernel_zero(double* rhs, int n0, int n1)
{
   int tid = threadIdx.x + blockIdx.x*blockDim.x;
   int tnum = gridDim.x*blockDim.x;
   for(int k = tid; true; k += tnum)
   {
        int i = k%n0;
		int j = k/n0;
		if (j < n1)
		{
		  rhs[j*n0+i] = 0.0;
		}
		else
		{
		   break;
		}
   }
}
 

bool cuda_zero(cudaDeviceProp dev, cudaStream_t* cudaStreams, double* rhs, int n0, int n1)
{
    dim3 gridDim;
	dim3 blockDim;
    int task = n0*n1;
    int maxth = dev.multiProcessorCount*dev.maxThreadsPerMultiProcessor;
    int tpt = (task-1)/maxth + 1;
    int dtn = (task-1)/tpt + 1;
    int avt = 512;
    int dtpb = min(dtn, avt);
    blockDim = dim3(dtpb);
    int dbn = (dtn - 1)/blockDim.x + 1;
    gridDim = dim3(dbn);
    cudakernel_zero<<<gridDim, blockDim, 0, cudaStreams[0]>>>(rhs, n0, n1);
	if(cudaPeekAtLastError() != cudaSuccess)
	{
	   printf("error zero\n");
	   return true;
	}
	cudaStreamSynchronize(cudaStreams[0]);
	return false;
}