import time 
import torch 
import torch_npu 

def benchmark(): 
    embed_size = 40000000 
    embed_dim = 512 
    
    embed = torch.empty([embed_size, embed_dim], dtype=torch.bfloat16) 
    step = 1 
    for bs in range(8, 1024 + 1, 8): 
        bs *= step 

        warmup_count = 10 
        run_count = 10 
        input_ids = torch.randint(0, embed_size, [warmup_count, bs], dtype=torch.int64) 
        for i in range(warmup_count): 
            input_embeds = embed[input_ids[i]].npu() 
        input_ids = torch.randint(0, embed_size, [run_count, bs], dtype=torch.int64) 
        torch.npu.synchronize() 
        a = time.time() 
        for i in range(run_count): 
            input_embeds = embed[input_ids[i]].npu() 
        torch.npu.synchronize() 
        b = time.time() 
        print("Kernel test {} time: {} - bandwidth: {}".format(bs, (b - a) / run_count * 10 * 1000, bs * run_count * embed_dim * 2 * 10/ ((b - a) * 10 * 1000000000))) 

if __name__ == '__main__': 
    benchmark()