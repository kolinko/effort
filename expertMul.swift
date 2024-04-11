//
//  expertMul.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 11/04/2024.
//

import Foundation

func expertMul(v: VectorFloat, by: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, quant: Double = 0.25) {
    if !goQ8 {
        bucketMulFast(v: v, by: by, expNo: expNo, out: out, quant: quant)
    } else {
        expertMulQ8(v: v, by: by, expNo: expNo, out: out, quant: quant)
    }
}


