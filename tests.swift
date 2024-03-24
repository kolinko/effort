//
//  tests.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 24/03/2024.
//

/*
 
 Asserts are better than tests.
 
 */


import Foundation

func modelRunTests() {
    let v = Vector(from: [0.1, 0.22, 0.33, 0.11, -0.21, 2, -0.01, 0.02])
    assert(v.scalarAt(3)[0] == 0.11)
    v.sort()
    assert(v.test("v.sort()", mul: 100, val: [-0.21, -0.01, 0.02, 0.1, 0.11, 0.22, 0.33, 2.0]))
    
    /*
    let h1 = tokens[0]
    let h2 = tokens[1]
    let s = ScalarFloat(value:0)
    gpu.startCapture()
    gpu.eval()
    gpu.deploy("dot2", buffers: [h1, h2, s], ints:[4], threadCount: 1024, threadGroupSize: [1024, 1, 1])
    gpu.eval()
    print(s[0])
    var sum : Float16 = 0.0
    for i in 0..<h1.rows {
        sum += h1[i]// * h2[i]
    }
    print(sum)
    gpu.stopCapture()
    exit(0)*/
}
