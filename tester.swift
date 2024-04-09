//
//  tester.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 08/04/2024.
//

import Foundation
/*
func saveTest(s: String) {
    let hh = h.copy()
    gpu.eval()
    testSaver[0]["token:\(thisToken)"] = hh}
*/

var testCount = 0

func testVec(_ title: String, _ v: VectorFloat) {
    testCount += 1
    if goVerify {
        let tt = (testLoader[title] as! Vector).asFloat32()
//        assert(tt.cosineSimilarityTo(v) > 0.99)
    }
}


func testVec32(_ title: String, _ v: VectorFloat) {
    testCount += 1

    if goVerify {
        let tt = testLoader[title] as! VectorFloat
        print(title, tt.cosineSimilarityTo(v))
  //      assert(tt.cosineSimilarityTo(v) > 0.99)
    }
}

func testReport(_ cond: Bool) {
    if goVerify && cond {
        print("✅ tests completed: \(testCount)")
        exit(0)
    }
}
