//
//  benchmark.swift
//  mul_col
//
//  Created by Tomasz Kolinko on 13/04/2024.
//

import Foundation

func testABCD(_ _query: String) {
    let query = "<s>[INST]\(_query)[/INST] The answer is number:"
    let logits = [28740, 28750, 28770, 28781]
    let embeded = t.embed(query)
    for effort in [1, 0.5, 0.4, 0.3, 0.25, 0.22, 0.20, 0.15, 0.10, 0.08] {
        let reply = runNetwork(tokens: embeded, effort:effort, limitLogits: logits).reply
        print(reply)
        print("\(Int(effort*100))% -> \(reply)")
    }
}

func verifyABCD(_ _query: String, answer: Int, scale: [Double]) -> [Bool] {
    let query = "<s>[INST]\(_query)[/INST] The answer is number:"
    let logits = [28740, 28750, 28770, 28781]
    let embeded = t.embed(query)
    var outputs = [Bool]()
    for effort in scale {
        let reply = runNetwork(tokens: embeded, effort:effort, limitLogits: logits).reply
        let score = Int(reply) == answer
        outputs.append(score)
    }
    return outputs
}

func goTruthhful() {
    var scale = [1, 0.5, 0.4, 0.3, 0.25, 0.22, 0.20, 0.15, 0.10, 0.08]
    var qa = [(String, Int)]()
    qa.append(("Is Sky blue?", 1))
    qa.append(("Is 4024-4011 == 4013?", 2))
    qa.append(("Is 4024-4011 == 13?", 1))

    var results = [[Bool]]()
    var count = 0
    for test in qa {
        count += 1
        print("Testing QA, \(count) of \(qa.count)")
        let prompt = "\(test.0)? Answer 1 for TRUE, 2 for FALSE"
        let out = verifyABCD(prompt, answer: test.1, scale: scale)
        print(out)
        results.append(out)
    }
    
    var result = Array(repeating: 0, count: scale.count)
    for r in results {
        for s in 0..<scale.count {
            result[s] += r[s] ? 1 : 0
        }
    }
    
    print("truthful replies")
    for s in 0..<scale.count {
        print("\(Int(scale[s]*100))% -> \(result[s])")
    }
}



func goBenchmarkSimilarity() {
    numTokens = 500
    let query = "[INST]Write a condensed perl implementations of the following: Dijkstra's algorithm, text search and quicksort. No comments, just write the code. Include data loaders.[/INST]"
//  let query = "[INST]Write an extremely long and convoluted story about potatoes[/INST]"

    let tokIds = encode(prompt: query)
    let baseline = runNetwork(isTest: true, tokens: t.embed(tokIds), effort: 1, srcTokenIds: tokIds)

    let tokenIds2 = baseline.hitMiss
    let control = runNetwork(isTest: true, tokens: t.embed(tokenIds2), effort: 1).hitMiss
    
    for effort in [1, 0.01, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.20, 0.15, 0.10, 0.08, 0.06, 0.04, 0.02] {
        let test = runNetwork(isTest: true, tokens: t.embed(tokenIds2), effort: effort).hitMiss
        var num = 0
        for i in 0..<control.count {
            num += control[i] == test[i] ? 1 : 0
        }
        let res = Double(100*Float(num)/Float(control.count))
        print("\(Int(effort*100))% -> " + "\(res, precision: 2)%")
        
    }
  
    exit(0)
}
