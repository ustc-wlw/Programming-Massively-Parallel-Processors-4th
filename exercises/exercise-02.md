![](C:\Users\mount.wang\AppData\Roaming\marktext\images\2024-07-22-14-25-22-image.png)

i=blockIdx.xblockDim.x + threadIdx.x;

![](C:\Users\mount.wang\AppData\Roaming\marktext\images\2024-07-22-14-28-14-image.png)

i=(blockIdx.xblockDim.x + threadIdx.x)2;

![](C:\Users\mount.wang\AppData\Roaming\marktext\images\2024-07-22-14-37-45-image.png)

<img src="file:///C:/Users/mount.wang/AppData/Roaming/marktext/images/2024-07-22-14-38-14-image.png" title="" alt="" width="699">

![](C:\Users\mount.wang\AppData\Roaming\marktext\images\2024-07-22-14-32-26-image.png)

8192

![](C:\Users\mount.wang\AppData\Roaming\marktext\images\2024-07-22-14-33-28-image.png)

v * sizeof(int)

![](C:\Users\mount.wang\AppData\Roaming\marktext\images\2024-07-22-14-34-57-image.png)

(void **) &A_d

![](C:\Users\mount.wang\AppData\Roaming\marktext\images\2024-07-22-14-36-25-image.png)

cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);

![](C:\Users\mount.wang\AppData\Roaming\marktext\images\2024-07-22-14-37-00-image.png)

cudaError_t err;
