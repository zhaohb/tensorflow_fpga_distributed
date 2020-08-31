/**********************************************
function: 2D Convolution of CNN
**********************************************/
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void Conv2D( __global float * restrict image_in,  //image input
                      __global float * restrict filter_in, //filter input
                               int filter_kernel_width,		    //filter kernel width
                               int filter_kernel_height,		    //filter kernel height
                               int filter_kernel_depth,		    //filter kernel depth
                               int filter_kernel_num,		    //filter kernel num
                               int image_width,
                               int image_height,
                               int image_num,
                               int image_pad_height,
                               int image_pad_width,
                               int image_new_height,
                               int image_new_width,
                               int conv_width_step,		    //stride cols
                               int conv_height_step,		    //stride rows
                      __global float * restrict image_out) //feature map output
{

    int x;		 //output_cols index 
	int y;		 //output_rows index
    int z;       //output_channel index
	int n, c, h, w;	 //filter coordinate,(kj, ki, kn)
	x = get_global_id(0);
	y = get_global_id(1);
	z = get_global_id(2) / image_num;

//    image_out[curIndex] = 0.0;
    int curIndex, tmpIndex, tmpFdex;
    int pad_rows = image_height + image_pad_height;
    int pad_cols = image_width + image_pad_width;
    n = get_global_id(2) % image_num;
    curIndex =  n * image_new_height * image_new_width * filter_kernel_num + y * image_new_width * filter_kernel_num + x * filter_kernel_num + z;
    image_out[curIndex]=0.0;
    #pragma unroll 1
    for(c = 0; c < filter_kernel_depth; c++) {
        #pragma unroll 1
        for(h = 0;h < filter_kernel_height;h++) {
            #pragma unroll 2
            for(w = 0; w < filter_kernel_width; w++) {
                int cols_val = x * conv_width_step + w;
                int rows_val = y * conv_height_step + h;
                tmpIndex = n * filter_kernel_depth * pad_rows * pad_cols + c * pad_rows * pad_cols + rows_val * pad_cols + cols_val;
                tmpFdex = z * filter_kernel_depth * filter_kernel_height * filter_kernel_width + c * filter_kernel_height * filter_kernel_width + h * filter_kernel_width + w;
                image_out[curIndex] = image_out[curIndex] + image_in[tmpIndex] * filter_in[tmpFdex];
            }            
        }        
    }    
    
}

__kernel void Conv2D_BackProp( __global float * restrict image_in,  //image input
                      __global float * restrict filter_in, //filter input
                               int filter_rows,		    //filter kernel width
                               int filter_cols,		    //filter kernel height
                               int input_depth,		    //filter kernel depth
                               int output_batch,		    //filter kernel num
                               int output_channel,
                               int output_rows,
                               int output_cols,
                               int pad_rows,
                               int pad_cols,
                      __global float * restrict image_out) //feature map output
{
	int x;		 //output_channel index 
	int y;		 //output_rows index
    int z;       //output_cols index
	int n, c, h, w;	 //filter coordinate,(kj, ki, kn)

	x = get_global_id(0)/output_batch;
	y = get_global_id(1);
	z = get_global_id(2);
//    image_out[curIndex] = 0.0;
    int curIndex, tmpIndex, tmpFdex;
    n = get_global_id(0) % output_batch;
        curIndex = n * output_channel * output_rows * output_cols + y * output_channel * output_cols + z * output_channel +x;
        image_out[curIndex] = 0.0;
        #pragma unroll 1
        for(c = 0; c < input_depth; c++) {
            #pragma unroll 1
            for(h = y; h < (y + filter_rows); h++ ) {
                #pragma unroll 2
                for(w = z; w < (z + filter_cols); w++) {
                    tmpIndex = n * input_depth * pad_rows * pad_cols + c * pad_rows * pad_cols + h * pad_cols + w;
                    tmpFdex = x * input_depth * filter_rows * filter_cols + c * filter_rows * filter_cols + (h - y) * filter_cols + w - z;
                    image_out[curIndex] = image_out[curIndex] + image_in[tmpIndex] * filter_in[tmpFdex];
        //       printf("x: %d, y: %d, z: %d, n: %d, c: %d, h: %d, w: %d, tmpIndex: %d, image_in[tmpIndex]: %f, tmpFdex: %d, filter_in[tmpFdex]: %f, curIndex: %d, image_out[curIndex]: %f\n",x,y,z,n,c,h,w,tmpIndex,image_in[tmpIndex],tmpFdex,filter_in[tmpFdex],curIndex, image_out[curIndex]); 
                }
            
            }
        
        }
    
       
}

__kernel void Conv2D_BackProp_Filter( __global float * restrict data_in,  //image input
                               __global float * restrict data_filter, //filter input
                                        int filter_rows,         //filter kernel width
                                        int filter_cols,         //filter kernel height
                                        int input_depth,         //filter kernel depth
                                        int out_batch,           //filter kernel num
                                        int out_dep,
                                        int pad_rows_out,
                                        int pad_cols_out,
                                        int pad_rows_input,
                                        int pad_cols_input,          //conv width step
                               __global float * restrict data_out) //feature map output
{  
    int x;       //global id x, correpsonding to out_dep 
    int y;       //global id y, corresponding to filter_rows
    int z;       //global id z, corresponding to filter_cols
    x = get_global_id(0);
    y = get_global_id(1);
    z = get_global_id(2);
    int n,c,h,w,b,i,j;
    for(n = 0; n < input_depth; n++) {
        c = x;
        h = y;
        w = z;
        int curIndex = n * out_dep * filter_rows * filter_cols + c * filter_rows * filter_cols + h * filter_cols + w;
        data_out[curIndex] = 0.0;
        for(b = 0; b < out_batch; b++) {
            for(i = h; i < (h + pad_rows_out); i++) {
                for(j = w; j < (w + pad_cols_out); j++) {
                    int index = n * out_batch * pad_rows_input * pad_cols_input + b * pad_rows_input * pad_cols_input + i * pad_cols_input + j;
                    int cIndex = c * out_batch * pad_rows_out * pad_cols_out + b * pad_rows_out * pad_cols_out + (i - h) * pad_cols_out + (j - w);
                    data_out[curIndex] = data_out[curIndex] + data_in[index] * data_filter[cIndex];
                }
            }
        }
    }
} 


__kernel void MaxPool( __global float * restrict image_in, 
                               int filter_rows,
                               int filter_cols,
                               int input_batch,		
                               int input_depth,		 
                               int input_rows,
                               int input_cols,
                               int pad_rows,
                               int pad_cols,
                               int stride_rows,		  
                               int stride_cols,		   
                               int new_rows,
                               int new_cols,
                      __global float * restrict out) 
{
    int pad_left = pad_cols / 2;
    int pad_top = pad_rows / 2;

    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2)/input_batch;
    int n,c,h,w;
//    for(n = 0; n < input_batch; n++) {
    n = get_global_id(2) % input_batch;
        c=z;
        float total = 0;
        int num = 0;
        float maxVal = 0.0;
       int curIndex = n*input_depth*new_rows*new_cols+y*new_cols*input_depth+x*input_depth+c;
       out[curIndex]=maxVal;
        #pragma unroll 1
        for(h = 0; h < filter_rows; h++) {
            #pragma unroll 2
            for(w = 0; w < filter_cols; w++) {
                int cols_val = x * stride_cols + w;
                int rows_val = y * stride_rows + h;
                if(cols_val < pad_left || cols_val > (input_cols + pad_left - 1) || rows_val < pad_top || rows_val > (input_rows + pad_top - 1)) {
                    maxVal=maxVal; 
                } else {
                    float curVal = image_in[n*input_depth*(input_rows+pad_rows)*(input_cols+pad_cols)+c*(input_rows+pad_rows)*(input_cols+pad_cols)+rows_val*(input_cols+pad_cols)+cols_val];
                    if(num == 0) {
                        maxVal = curVal;
                    } else {
                        maxVal = maxVal < curVal ? curVal : maxVal;
                    }
                    num=num+1;
                } 
            }
        }
        out[curIndex]=maxVal;        
//    }
}


__kernel void AvgPool( __global float * restrict image_in,  //image input
                               int filter_rows,		    
                               int filter_cols,		    
                               int input_batch,		    
                               int input_depth,		    
                               int input_rows,
                               int input_cols,
                               int pad_rows,
                               int pad_cols,
                               int stride_rows,		    
                               int stride_cols,		    
                               int new_rows,
                               int new_cols,
                      __global float * restrict out) 
{
    int pad_left = pad_cols / 2;
    int pad_top = pad_rows / 2;

    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2) / input_batch;
    int n,c,h,w;
//    for(n = 0; n < input_batch; n++) {
    n = get_global_id(2) % input_batch;
        c=z;
        float total = 0;
        int num = filter_cols * filter_rows;
        #pragma unroll 1
        for(h = 0; h < filter_rows; h++) {
            #pragma unroll 2
            for(w = 0; w < filter_cols; w++) {
                int cols_val = x * stride_cols + w;
                int rows_val = y * stride_rows + h;
                if(cols_val < pad_left || cols_val > (input_cols + pad_left - 1) || rows_val < pad_top || rows_val > (input_rows + pad_top - 1)) {
                    --num;
                } else {
                    total = total + image_in[n*input_depth*(input_rows+pad_rows)*(input_cols+pad_cols)+c*(input_rows+pad_rows)*(input_cols+pad_cols)+rows_val*(input_cols+pad_cols)+cols_val];                    
                } 
            }
        }
        int index = n*input_depth*new_rows*new_cols+y*input_depth*new_cols+x*input_depth+z;
        out[index] = total / (float)num;   
//    }
}

__kernel void DeepthMaxPool( __global float * restrict image_in,
                               int new_batch,
                               int new_depth,		    
                               int new_rows,
                               int new_cols,
                               int stride_depth,		    
                      __global float * restrict out) //feature map output
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2);
    int m = 0;
    int input_depth = stride_depth * new_depth;
    #pragma unroll 1
    while(m < new_depth) {
        float maxVal = image_in[x*new_rows*new_cols*input_depth+y*new_cols*input_depth+z*input_depth+m*stride_depth];
        int index = 1;
        #pragma unroll 1
        while(index < stride_depth) {
            float curData = image_in[x*new_rows*new_cols*input_depth+y*new_cols*input_depth+z*input_depth+m*stride_depth+index];
           // printf("maxVal %f,curData %f\n",maxVal,curData);
            maxVal = (maxVal > curData) ? maxVal : curData;
            index++;
        }
        out[x*new_rows*new_cols*new_depth+y*new_cols*new_depth+z*new_depth+m]=maxVal;
        //std::cout<<"new batch: "<<new_batch<<" new depth: "<<new_depth<<" new rows: "<<new_rows<<" new_cols: "<<new_cols<<" stride_depth: "<<stride_depth<<" x:"<<x<<" y:"<<y<<" z"<<z<<" m:"<<m<<" max:"<<max<<std::endl;
   //     printf("new batch : %d new depth: %d new rows: %d new cols: %d stride_depth: %d x: %d y: %d z: %d m: %d max: %f\n",new_batch,new_depth,new_rows,new_cols,stride_depth,x,y,z,m,maxVal);
        m++;
    }
}


__kernel void relu( __global float * restrict data_in, 
                    __global float * restrict data_out,
                                       int flag,
                                       int input_num,		
                                       int input_height,		 
                                       int input_width,
                                       int input_channel ) 
{ 
    int x = get_global_id(0);
    int y = get_global_id(1);
    if(flag == 1) {
        int index = x * input_num * input_width * input_channel + y * input_num * input_channel;
        int end = input_num * input_channel;
        double d = 0.0;
        int i;
        #pragma unroll 2
        for(i = 0;i < end; i++) {
            d = (double) data_in[i+index];
            if(d > 0) {
                data_out[i+index] = d;
            } else {
                data_out[i+index] = 0.0f;
            }
        
        }
    } else {
        int index = x * input_height + y;
        double d = 0.0;
        int end = input_height;
        int i;
        #pragma unroll 1
        for(i = 0; i < end; i++) {
            d = (double) data_in[i + index];
            if(d > 0) {
                data_out[i+index] = d;
            } else {
                data_out[i+index]= 0;            
            }        
        } 
    }
}
__kernel void MaxPoolGrad( __global float * restrict image_in, 
                               int filter_rows,
                               int filter_cols,
                               int input_batch,		
                               int input_depth,		 
                               int input_rows,
                               int input_cols,
                               int pad_rows,
                               int pad_cols,
                               int stride_rows,		  
                               int stride_cols,		   
                               int new_rows,
                               int new_cols,
                      __global float * restrict out,
                      __global float * restrict op) 
{
    int pad_left = pad_cols / 2;
    int pad_top = pad_rows / 2;

    
    int x = get_global_id(0);     //new cols
    int y = get_global_id(1);     // new rows
    int z = get_global_id(2)/input_batch; // input_depth
    int n,c,h,w;
    int lh,lw;
//    for(n = 0; n < input_batch; n++) {
    n = get_global_id(2) % input_batch;    //input_batch
        c=z;
        float total = 0;
        int num = 0;
        float maxVal = 0.0;
       int curIndex = n*input_depth*new_rows*new_cols+y*new_cols*input_depth+x*input_depth+c;
       //out[curIndex]=maxVal;
        #pragma unroll 1
        for(h = 0; h < filter_rows; h++) {
            #pragma unroll 2
            for(w = 0; w < filter_cols; w++) {
                int cols_val = x * stride_cols + w;
                int rows_val = y * stride_rows + h;
                if(cols_val < pad_left || cols_val > (input_cols + pad_left - 1) || rows_val < pad_top || rows_val > (input_rows + pad_top - 1)) {
                    maxVal=maxVal; 
                } else {
                    float curVal = image_in[n*input_depth*(input_rows+pad_rows)*(input_cols+pad_cols)+c*(input_rows+pad_rows)*(input_cols+pad_cols)+rows_val*(input_cols+pad_cols)+cols_val];
                    if(num == 0) {
                        maxVal = curVal;
                        lw = cols_val - pad_left;
                        lh = rows_val - pad_top;

                    } else {
                        lw = curVal > maxVal ? (cols_val-pad_left) : lw;
                        lh = curVal > maxVal ? (rows_val - pad_top) : lh;
                        maxVal = curVal > maxVal ? curVal : maxVal;
                        
                    }
                    num=num+1;
                } 
            }
        }
        int outIndex = n * input_depth * input_rows * input_cols + lh * input_cols * input_depth + lw * input_depth + c;
        out[outIndex] = op[curIndex];        
//    }
}

__kernel void AvgPoolGrad( __global float * restrict image_in,  //image input
                               int filter_rows,		    
                               int filter_cols,		    
                               int input_batch,		    
                               int input_depth,		    
                               int input_rows,
                               int input_cols,
                               int pad_rows,
                               int pad_cols,
                               int stride_rows,		    
                               int stride_cols,		    
                               int new_rows,
                               int new_cols,
                          __global float * restrict out) 
{
    int pad_left = pad_cols / 2;
    int pad_top = pad_rows / 2;

    
    int x = get_global_id(0);
    int y = get_global_id(1);
    int z = get_global_id(2) / input_batch;
    int n,c,h,w;
//    for(n = 0; n < input_batch; n++) {
    n = get_global_id(2) % input_batch;
    c=z; 
    int num = filter_cols * filter_rows;
    int arr[100]; 
    int curNum = 0; 
    #pragma unroll 1
    for(h = 0; h < filter_rows; h++) {
        #pragma unroll 2
        for(w = 0; w < filter_cols; w++) {
            int cols_val = x * stride_cols + w;
            int rows_val = y * stride_rows + h;
            if(cols_val < pad_left || cols_val > (input_cols + pad_left - 1) || rows_val < pad_top || rows_val > (input_rows + pad_top - 1)) {
                --num;
            } else {
                int curIndex = n * input_depth * input_rows * input_cols + (rows_val - pad_top) * input_cols * input_depth + (cols_val - pad_left) * input_depth + c;
                arr[curNum] = curIndex;
                curNum++;
            } 

        }
    }

    int index = n*input_depth*new_rows*new_cols+y*input_depth*new_cols+x*input_depth+z;


    int i;
  #pragma unroll 2
    for(i =0 ;i < num; i++) {
        int j = arr[i];
        out[j] = image_in[index] / num;

    }
}




