//
//  tester.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 08/04/2024.
//

import Foundation

private var testCount = 0
private let testVer = "1.0" + "-" + (goQ8 ? "Q8" : "FP16")

private let testLoader = TensorLoader(path: "./", model: "tests-\(testVer)")
private let testSaver = TensorSaver(path: "./", model: "tests-\(testVer)")

func testVec(_ title: String, _ v: VectorFloat) {
    testCount += 1
    if goSaveTests {
        let hh = v.copy()
        gpu.eval()
        testSaver[0][title] = hh
    } else if goVerify {
        let tt = testLoader[title] as! VectorFloat
        let score = tt.cosineSimilarityTo(v)
        assert(score > 0.99, "Error in test \(testCount): \(title); \(score)")
    }
}


func testVec32(_ title: String, _ v: VectorFloat) {
    testCount += 1
    if goSaveTests {
        let hh = v.copy()
        gpu.eval()
        testSaver[0][title] = hh
    } else if goVerify {
        let tt = testLoader[title] as! VectorFloat
        let score = tt.cosineSimilarityTo(v)
        print(title, score)
        assert(score > 0.99, "Error in test \(testCount): \(title); \(score)")
    }
}

func testReport(_ cond: Bool) {
    if !cond {return}
    if goSaveTests {
        testSaver.save()
        print("☑️ tests saved: \(testCount)")
        exit(0)
    } else if goVerify {
        print("✅ tests completed: \(testCount)")
        exit(0)
    }
}
