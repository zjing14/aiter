import os
import sys
from dataclasses import dataclass
import copy
from pathlib import Path
import pandas as pd
import argparse
import shutil

class get_all_instances:
    def __init__(self):
        self.mn_tile = [16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256]
        self.k_tile = [64, 128, 256, 512]
        self.block_size = [64, 128, 256]
        self.mn_warp = [1, 2, 4]
        #self.pipeline = [1, 2, 3, 4, 5]
        self.pipeline = [1, 2, 3, 4]
        self.scheduler = ["Intrawave", "Interwave"]
        self.nbyte_a = 1
        self.nbyte_b = 1
        self.nbyte_c = 2
        self.kpack = 128 // (self.nbyte_a * 8)

    def is_valid(self, blk, m, n, k, m_warp, n_warp, pipeline, scheduler):
        num_warps = blk // 64

        if(m_warp * n_warp != num_warps):
            return False

        #alignment
        if(m % (m_warp * 16) != 0 or n % (n_warp * 16) != 0):
            return False

        if(k % self.kpack != 0):
            return False

        k0 = k // self.kpack
        if((m * k0) % blk != 0 or (n * k0) % blk != 0):
            return False

        #get LDS usage for a/b tile, pipeline v4 has double buffer
        lds_a = m * k * self.nbyte_a * (2 if pipeline == 4 else 1)
        lds_b = n * k * self.nbyte_b * (2 if pipeline == 4 else 1)
        lds_c = m * n * self.nbyte_c

        #lds size must no more than 64KB
        if((lds_a + lds_b) > 64 * 1024):
            return False

        #register usage per thread for a/b tile buffer must be no more than 512 * 4 bytes
        if((lds_a + lds_b + lds_c) // blk > 512 * 4):
            return False

        #calculate occupancy based on LDS and register usage
        occupancy = 64 * 1024 // (lds_a + lds_b)
        occupancy = min(occupancy, 256 * 4 // ((lds_a + lds_b) // blk))
        occupancy = min(occupancy, 256 * 4 // (lds_c // blk))


        #Interwave requires occupancy > 1
        if(occupancy < 2 and scheduler == "Interwave"):
            return False

        #occupancy require by pipeline v3
        occupancy_min_v3 = 2 if ((m * n // blk) <= 128) else 1
        if(occupancy < occupancy_min_v3 and pipeline == 3):
            return False

        if(not(m == 256 and n == 256)):
            return False
        #m_per_warp = m // m_warp
        #n_per_warp = n // n_warp

        #m_repeat = m_per_warp // 16
        #n_repeat = n_per_warp // 16

        #if(abs(m_repeat - n_repeat) > 6):
            #return False

        #if((m < 128 and n < 128) and k < 128):
            #return False

        #if(m == 256 and n == 256 and pipeline < 3):
            #return False

        #if((m < 128 or n < 128) and pipeline > 3):
            #return False

        #if((m > 128 and n > 128) and pipeline < 3):
            #return False

        #if(scheduler == "Interwave" and pipeline > 2):
            #return False

        #num_mfma = (m * n) // (16 * 16)
        #if((num_mfma * 64) % blk != 0):
            #return False

        #if((num_mfma * 64) // blk > 64):
            #return False
        #if((m // 16 > 16) or (n // 16 > 16)):
            #return False

        return True

    def get_mfma(self, blk, m, n, m_warp, n_warp):
        m_per_warp = m // m_warp
        n_per_warp = n // n_warp


        # use 32x32 mfma if possible
        if(m_per_warp % 32 == 0 and n_per_warp % 32 ==0):
            mfma = 32
        else:
            mfma = 16

        m_repeat = m_per_warp // mfma
        n_repeat = n_per_warp // mfma

        return [mfma, mfma, m_repeat, n_repeat]

    def try_c_transfer(self, blk, m_warp, n_warp, mfma, m_repeat, n_repeat, n_vec):
        c_shuffle_m = m_warp * mfma * m_repeat
        c_shuffle_n = n_warp * mfma * n_repeat

        tid_n = c_shuffle_n // n_vec
        tid_m = blk // tid_n
        return tid_m, tid_n


    def get_c_transfer(self, blk, m, n, m_warp, n_warp, mfma_cfg):
        mfma, _, m_repeat, n_repeat = mfma_cfg

        n_vec = 128 // (8 * self.nbyte_c)

        c_m_repeat = 1
        c_n_repeat = 1

        tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma, c_m_repeat, c_n_repeat, n_vec)

        ctgs_store_size = n_vec * tid_n * self.nbyte_c

        #if possible, enlarge c_n_repeat to fit 128B cacheline
        if(ctgs_store_size < 128 and n_repeat % 2 == 0):
            c_n_repeat = 2
            _tid_m, _tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma, c_m_repeat, c_n_repeat, n_vec)
            if(_tid_m * _tid_n == blk):
                return [[c_m_repeat, c_n_repeat], [1, _tid_m, 1, _tid_n], [n_vec, n_vec, 1]]

        #if not meet, try enlarge c_m_repeat
        if(tid_m * tid_n != blk):
            c_m_repeat = 2
            tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma, c_m_repeat, c_n_repeat, n_vec)
            if(tid_m * tid_n == blk):
                return [[c_m_repeat, c_n_repeat], [1, _tid_m, 1, _tid_n], [n_vec, n_vec, 1]]

        #if not meet, try reduce vec_len
        if(tid_m * tid_n != blk):
            n_vec = 4
            tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma, c_m_repeat, c_n_repeat, n_vec)
            if(tid_m * tid_n == blk):
                return [[c_m_repeat, c_n_repeat], [1, _tid_m, 1, _tid_n], [n_vec, n_vec, 1]]

        if(tid_m * tid_n != blk):
            raise Exception("cannot find proper cshuffle")

        return [[c_m_repeat, c_n_repeat], [1, tid_m, 1, tid_n], [n_vec, n_vec, 1]]

    def get_ab_transfer(self, blk, mn, k):
        k0 = k // self.kpack
        tid_k0 = k0
        tid_mn = blk // tid_k0

        return [tid_k0, tid_mn, 1]

    def gen_list(self):
        num_i = 0

        for blk in self.block_size:
            for m in self.mn_tile:
                for n in self.mn_tile:
                    for k in self.k_tile:
                        for m_warp in self.mn_warp:
                            for n_warp in self.mn_warp:
                                for pipeline in self.pipeline:
                                    for scheduler in self.scheduler:
                                        if(self.is_valid(blk, m, n, k, m_warp, n_warp, pipeline, scheduler)):
                                            try:
                                                mfma_cfg = self.get_mfma(blk, m, n, m_warp, n_warp)
                                                a_load = self.get_ab_transfer(blk, m, k)
                                                b_load = self.get_ab_transfer(blk, n, k)
                                                c_shuffle = self.get_c_transfer(blk, m, n, m_warp, n_warp, mfma_cfg)
                                                print(f"{num_i:>4}: kernelInstance({blk:>4},\t{m:>4},\t{n:>4},\t{k:>4},\t{mfma_cfg[0]:>4},\t{mfma_cfg[1]:>4},\t{mfma_cfg[2]:>2},\t{mfma_cfg[3]:>2},\t{a_load},\t{b_load},\t{c_shuffle[1]},\t{c_shuffle[2]},\t{c_shuffle[0][0]},\t{c_shuffle[0][1]},\t\"{scheduler}\",\t{pipeline}),")
                                                num_i += 1
                                            except Exception as e:
                                                print(f"cannot generate proper instance, e = {e}")
        print(f"total instance = {num_i}")


if __name__ == "__main__":
    gen_instance_list = get_all_instances()
    gen_instance_list.gen_list()
