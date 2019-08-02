#include "heap.h"

using namespace std;

void top_k(vector<point>& input_arr, int32_t n, int32_t k) {
    // O(k)
    // we suppose the k element of the min heap if the default top k element
    min_heap_t min_heap(input_arr, k);
    min_heap.build_heap_from_bottom_to_top();
    
    for (int32_t i = k; i < n; ++i) {
        // compare each element with the min element of the min heap
        // if the element > the min element of the min heap
        // we think may be the element is one of what we wanna to find in the top k
        if (input_arr[i].semi > min_heap.arr[0].semi){
            // swap
            min_heap.arr[0] = input_arr[i];
            
            // heap adjust
            min_heap.heap_adjust_from_top_to_bottom(0, k - 1);
        }
    }
    
    input_arr.assign(min_heap.arr.begin(),min_heap.arr.end());
}

void top_k_with_NMS(vector<point>& input_arr, int32_t n, int32_t k, int32_t nms_thresh) {
    // build  heap first
    max_heap_t max_heap(input_arr, n);
    max_heap.build_heap_from_bottom_to_top();
    
    // sort top k
    point tmp;
    for (int32_t i = n - 1; i >= max_heap.arr.size() - k;) {
        // move heap top to end
        tmp = max_heap.arr[0];
        max_heap.arr[0] = max_heap.arr[i];
        max_heap.arr[i] = tmp;
        
        for (int32_t j = i+1; j < n; j++)
        {
            if(abs(max_heap.arr[i].H-max_heap.arr[j].H)<=nms_thresh && abs(max_heap.arr[i].W-max_heap.arr[j].W)<=nms_thresh)
            {
                max_heap.arr.erase(max_heap.arr.begin()+i);
                break;
            }
        }

        // adjust the heap
        max_heap.heap_adjust_from_top_to_bottom(0, --i);
    }
    
    input_arr.assign(max_heap.arr.end()-k,max_heap.arr.end());
}