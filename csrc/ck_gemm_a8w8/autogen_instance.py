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
        self.k_tile = [32, 64, 128, 256, 512]
        self.block_size = [64, 128, 256]
        self.mn_warp = [1, 2, 4]
        self.nbyte_a = 2
        self.nbyte_b = 2
        self.nbyte_c = 2
        self.kpack = 128 // (self.nbyte_a * 8)

    def is_valid(self, blk, m, n, k, m_warp, n_warp):
        valid = True

        num_warp = blk // 64

        if(m_warp * n_warp != num_warp):
            return False

        if(m % (m_warp * 16) != 0):
            return False

        if(n % (n_warp * 16) != 0):
            return False

        lds_a = m * k * self.nbyte_a
        lds_b = n * k * self.nbyte_b
        if((lds_a + lds_b) > 64 * 1024):
            return False
        if((lds_a + lds_b) > 128 * 4 * blk):
            return False

        num_mfma = (m * n) // (16 * 16)
        if((num_mfma * 64) % blk != 0):
            return False
        if((num_mfma * 64) // blk > 64):
            return False

        if(k % self.kpack != 0):
            return False
        k0 = k // self.kpack
        if((m * k0) % blk != 0 or (n * k0) % blk != 0):
            return False

        return valid

    def get_mfma(self, blk, m, n, m_warp, n_warp):
        m_per_warp = m // m_warp
        n_per_warp = n // n_warp

        if(m_per_warp % 32 == 0 and n_per_warp % 32 ==0):
            mfma = 32
        else:
            mfma = 16

        m_repeat = m_per_warp // mfma
        n_repeat = n_per_warp // mfma

        return [mfma, mfma, m_repeat, n_repeat]

	#c_shuffle_lookup_table = {
		#(32, 4, 1, 1, 1): [2, 1, 64, 4, 8],
	#}

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

        if((tid_n < 4) and (n_repeat % 2 == 0)):
            c_n_repeat = 2
            tid_m, tid_n = self.try_c_transfer(blk, m_warp, n_warp, mfma, c_m_repeat, c_n_repeat, n_vec)

        return [[c_m_repeat, c_n_repeat], [1, tid_m, 1, tid_n], n_vec]

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
                                if(self.is_valid(blk, m, n, k, m_warp, n_warp)):
                                    mfma_cfg = self.get_mfma(blk, m, n, m_warp, n_warp)
                                    a_load = self.get_ab_transfer(blk, m, k)
                                    b_load = self.get_ab_transfer(blk, n, k)
                                    c_shuffle = self.get_c_transfer(blk, m, n, m_warp, n_warp, mfma_cfg)
                                    print(f"block_size = {blk} m_tile = {m} n_tile = {n} k_tile = {k} m_warp = {m_warp} n_warp = {n_warp} mfma = {mfma_cfg}  a_load = {a_load} b_load = {b_load} c_shuffle = {c_shuffle}")
                                    num_i += 1
        print(f"total instance = {num_i}")


if __name__ == "__main__":
    gen_instance_list = get_all_instances()
    gen_instance_list.gen_list()
