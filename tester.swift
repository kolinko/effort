//
//  tester.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 08/04/2024.
//

import Foundation

private var testCount = 0
private var driftCount = 0
private let testVer = "5.1.1" + "-" + (goQ8 ? "Q8" : "FP16") + (goMistral ? "mistral" : "mixtral") + ("-noLay\(numLayers)")

private let testLoader = TensorLoader(path: "./", model: "tests-\(testVer)")
private let testSaver = TensorSaver(path: "./", model: "tests-\(testVer)")

//rivate

private var testLog = [String]()

private func testTest(_ title: String, _ score: Float) {
    testLog.append("[\(testCount)] \(title); \(score)")
    if score < 0.99 {
        if title.contains("h_in") {
            print("❕Drift in \(testCount): \(title); \(score)❕")
            driftCount += 1
//            exit(1)
        } else {
            for i in min(0, testLog.count-20)..<testLog.count {
                print(testLog[i])
            }

//            assertionFailure("❌ error in \(testCount): \(title); \(score)")
        }
    }
}
//expr let x = ( testLoader["ovector:4"] as! VectorFloat); (0..<4096).map { ( x[$0], outputVector[$0] ) }
func getVec(_ title: String) -> VectorFloat {
    return testLoader[title] as! VectorFloat
}

func testVec(_ title: String, _ v: VectorFloat) {
    testCount += 1
    if goSaveTests {
        let hh = v.copy()
        gpu.eval()
        testSaver[0][title] = hh
    } else if goVerify {
        let tt = testLoader[title] as! VectorFloat
        let score = tt.cosineSimilarityTo(v)
        testTest(title, score)
    }
}

func cosVec(_ title: String, _ v: VectorFloat) -> Float {
    let tt = testLoader[title] as! VectorFloat
    let score = tt.cosineSimilarityTo(v)
    return score
}

func tv(_ title: String) -> VectorFloat {
    return testLoader["ovector:4"] as! VectorFloat
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
        testTest(title, score)
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
        print("    drifts: \(driftCount)")
        // exit(0)
    }
}
