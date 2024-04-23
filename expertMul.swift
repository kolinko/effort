//
//  expertMul.swift
//  effort
//
//

/*
 
 A wrapper for the multiplications.
 Useful for testing, to change the algorithm just here.
 
 It will probably be refactored away probably, but for now it's useful to have it here.
 
 */

import Foundation

private let tmpExpZero = ScalarFloat(value: 0)

func expertMul(v: VectorFloat, by: ExpertWeights, out: VectorFloat, effort: Double = 0.25) {
    expertMul(v: v, by: by, expNo: tmpExpZero, out: out, effort: effort)
}

func expertMul(v: VectorFloat, by: ExpertWeights, expNo: ScalarFloat, out: VectorFloat, effort: Double = 0.25) {
    if !goQ8 {
        bucketMul(v: v, by: by, expNo: expNo, out: out, effort: effort)
//        expertMulSlow(v: v, by: by, expNo: expNo, out: out, effort: effort)
    } else {
        assert(false, "not tested with the current iteration")
        expertMulQ8(v: v, by: by, expNo: expNo, out: out, effort: effort)
    }
}


