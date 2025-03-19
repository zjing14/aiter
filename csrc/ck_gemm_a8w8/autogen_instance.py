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
        self.pipeline = [1, 2, 3]
        self.scheduler = ["Intrawave"]
        self.nbyte_a = 1
        self.nbyte_b = 1
        self.nbyte_acc = 4
        self.nbyte_c = 2
        self.kpack = 128 // (self.nbyte_a * 8)

    def is_valid(self, blk, m, n, k, m_warp, n_warp, pipeline, scheduler):

        num_warps = blk // 64
        if(m_warp * n_warp != num_warps):
            return False

        # alignment with mfma
        if(m % (m_warp * 16) != 0 or n % (n_warp * 16) != 0):
            return False

        if(k % self.kpack != 0):
            return False

        # make sure full loads
        k0 = k // self.kpack
        if((m * k0) % blk != 0 or (n * k0) % blk != 0):
            return False

        # get LDS usage for a/b tile, pipeline v4 has double buffer
        lds_a = m * k * self.nbyte_a * (2 if pipeline > 1 else 1)
        lds_b = n * k * self.nbyte_b * (2 if pipeline > 1 else 1)
        lds_c = m * n * self.nbyte_c

        # lds size must no more than 64KB
        if((lds_a + lds_b) > 64 * 1024):
            return False

        reg_a = m * k * self.nbyte_a // blk
        reg_b = n * k * self.nbyte_b // blk
        reg_c = m * n * self.nbyte_acc // blk

        # register usage for a/b tile buffer must be no more than 256 * 4 bytes
        if((reg_a + reg_b)  > 256 * 4):
            return False

        # register usage for a/b/c tile buffer must be no more than 512 * 4 bytes
        if((reg_a + reg_b + reg_c)  > 512 * 4):
            return False

        #calculate occupancy based on LDS and register usage
        occupancy = 64 * 1024 // (lds_a + lds_b)
        occupancy = min(occupancy, 256 * 4 // (reg_a + reg_b + reg_c))

        #Interwave requires occupancy >= 2
        if(occupancy < 2 and scheduler == "Interwave"):
            return False

        if(pipeline > 2 and scheduler == "Interwave"):
            return False

        return True

    def is_good(self, blk, m, n, k, m_warp, n_warp, pipeline, scheduler):
        #if(not (blk == 256 and m == 32 and n == 64)):
            #return False

        m_per_warp = m // m_warp
        n_per_warp = n // n_warp

        m_repeats = m_per_warp // 32 if m_per_warp % 32 == 0 else m_per_warp // 16

        if(m_repeats < 4 and pipeline == 3):
            return False

        #limit warp workloads
        if((m_per_warp > 64 or n_per_warp > 64) and blk < 256):
            return False

        if((m < 128 or n < 128) and pipeline > 3):
            return False

        if((m < 128 and n < 128) and k < 256):
            return False

        if((m < 32 or n < 32) and pipeline > 2):
            return False

        if((m >= 64 and n >= 64) and pipeline < 3):
            return False

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

    def is_valid_c_transfer(self, blk, tid_m, tid_n, m_warp, n_warp, c_m_repeat, c_n_repeat, mfma_cfg, n_vec):
        mfma, _, m_repeat, n_repeat = mfma_cfg

        c_shuffle_m = m_warp * mfma * c_m_repeat
        c_shuffle_n = n_warp * mfma * c_n_repeat

        if(tid_m * tid_n != blk):
            return False

        if(c_shuffle_m % tid_m != 0):
            return False

        if(c_shuffle_n % (tid_n * n_vec) != 0):
            return False

        if(m_repeat % c_m_repeat != 0):
            return False

        if(n_repeat % c_n_repeat != 0):
            return False

        lds_c_shuffle = c_shuffle_m * c_shuffle_n * self.nbyte_acc

        if(lds_c_shuffle > 32 * 1024):
            return False

        return True


    def try_c_transfer(self, blk, m_warp, n_warp, mfma_cfg, c_m_repeat, c_n_repeat, n_vec):
        mfma, _, m_repeat, n_repeat = mfma_cfg

        c_shuffle_m = m_warp * mfma * c_m_repeat
        c_shuffle_n = n_warp * mfma * c_n_repeat

        # load n dim first
        tid_n = c_shuffle_n // n_vec
        tid_m = blk // tid_n

        if(self.is_valid_c_transfer(blk, tid_m, tid_n, m_warp, n_warp, c_m_repeat, c_n_repeat, mfma_cfg, n_vec)):
            return tid_m, tid_n
        else:
            return 0, 0

    def get_c_transfer(self, blk, m, n, m_warp, n_warp, mfma_cfg):
        mfma, _, m_repeat, n_repeat = mfma_cfg

        n_vec = 128 // (8 * self.nbyte_c)

        c_m_repeat = 1
        c_n_repeat = 1

        c_shuffle_n = n_warp * mfma * c_n_repeat

        ctgs_store_size = c_shuffle_n * self.nbyte_c

        #if possible, enlarge c_n_repeat to fit 128B cacheline
        if(ctgs_store_size < 128 and n_repeat % 2 == 0):
            c_n_repeat = 2
            tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma_cfg, c_m_repeat, c_n_repeat, n_vec)
            if(tid_n * tid_m == blk):
                return [[c_m_repeat, c_n_repeat], [1, tid_m, 1, tid_n], [n_vec, n_vec, 1]]

        #if not meet, try enlarge c_m_repeat
        if(m_repeat % 2 == 0):
            c_m_repeat = 2
            tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma_cfg, c_m_repeat, c_n_repeat, n_vec)
            if(tid_n * tid_m == blk):
                return [[c_m_repeat, c_n_repeat], [1, tid_m, 1, tid_n], [n_vec, n_vec, 1]]

        #if not meet, try reduce vec_len
        n_vec = n_vec // 2
        tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma_cfg, c_m_repeat, c_n_repeat, n_vec)
        if(tid_n * tid_m == blk):
            return [[c_m_repeat, c_n_repeat], [1, tid_m, 1, tid_n], [n_vec, n_vec, 1]]

        #still not meat, raise an Exception
        if(tid_n * tid_m != blk):
            raise Exception("cannot find proper cshuffle")

        return [-1]

    def get_ab_transfer(self, blk, mn, k):

        # load k dim first
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
                                        if(self.is_valid(blk, m, n, k, m_warp, n_warp, pipeline, scheduler) and self.is_good(blk, m, n, k, m_warp, n_warp, pipeline, scheduler)):
                                            try:
                                                mfma_cfg = self.get_mfma(blk, m, n, m_warp, n_warp)
                                                a_load = self.get_ab_transfer(blk, m, k)
                                                b_load = self.get_ab_transfer(blk, n, k)
                                                c_shuffle = self.get_c_transfer(blk, m, n, m_warp, n_warp, mfma_cfg)
                                                print(f"{num_i:>4}: kernelInstance({blk:>4},\t{m:>4},\t{n:>4},\t{k:>4},\t{mfma_cfg[0]:>4},\t{mfma_cfg[1]:>4},\t{mfma_cfg[2]:>2},\t{mfma_cfg[3]:>2},\t{a_load},\t{b_load},\t{c_shuffle[1]},\t{c_shuffle[2]},\t{c_shuffle[0][0]},\t{c_shuffle[0][1]},\t\"{scheduler}\",\t{pipeline}),")
                                                num_i += 1
                                            except Exception as e:
                                                print(f"cannot generate proper instance {blk, m, n, k, m_warp, n_warp, mfma_cfg}, e = {e}")
        print(f"total instance = {num_i}")


if __name__ == "__main__":
    gen_instance_list = get_all_instances()
    gen_instance_list.gen_list()
